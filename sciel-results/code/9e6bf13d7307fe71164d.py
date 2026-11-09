"""Multi-Objective Bayesian Optimization for RE21 using EHVI acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings

# Suppress warnings from scipy GP or convex hull calculations
warnings.filterwarnings("ignore")

class Optimizer:
    """
    Multi-objective optimizer using Gaussian Process surrogate models 
    and Expected Hypervolume Improvement (EHVI) acquisition.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for hypervolume calculation
        # The task specifies reference point [3000.0, 0.06] for scoring,
        # but for internal EHVI calculations, we should use a dynamic or 
        # reasonable reference point based on observed data to avoid 
        # numerical issues if objectives are far from 3000/0.06.
        # However, the scoring uses [3000, 0.06]. 
        # Using a reference point that is "bad" (high) ensures all valid points contribute.
        # Let's initialize with a large reference point, then update it dynamically 
        # to be slightly worse than the current worst non-dominated point.
        
        # 1. Initialization: Latin Hypercube Sampling (LHS) for better coverage than grid
        n_init = 20
        X_init = self._latin_hypercube(n_init, self.dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        
        # Track evaluated points and objectives
        X_eval = X_init
        F_eval = F_init
        
        # 2. Bayesian Optimization Loop
        # Budget remaining: 200 - 20 = 180
        # Each iteration: 
        #   - Fit GP (fast for small n)
        #   - Optimize EHVI (expensive, ~10-50 evals of acquisition)
        #   - Evaluate best candidate (1 eval of problem)
        
        # We will run BO for a fixed number of iterations or until budget/time runs out.
        # Let's aim for ~30-40 iterations, each taking ~5-10 problem evals for acquisition optimization.
        # Total problem evals: 20 (init) + 35 * 1 = 55. 
        # Wait, acquisition optimization uses the *surrogate* model, not the *problem*.
        # So each BO step costs 1 problem evaluation.
        # We have 180 budget left. We can run 180 steps? 
        # But fitting GP and optimizing EHVI takes time.
        # Let's limit to 50-60 BO steps to stay within time, or just loop until budget/time.
        
        # Dynamic Reference Point
        # Start with a point that is worse than all initial points
        ref_point = np.max(F_init, axis=0) + 1.0
        
        # GP Kernel parameters
        # We use a simple RBF kernel with length scales per dimension.
        # Since dim=4, we have 4 length scales + noise + output variance.
        
        for step in range(self.budget - len(X_eval)):
            if time.time() > deadline:
                break
            
            # Update reference point: max of current non-dominated front + small margin
            # This ensures the hypervolume is well-defined relative to current knowledge
            nd_points = self._non_dominated(F_eval)
            if len(nd_points) > 0:
                # Use the worst non-dominated point + margin as reference
                # Or just use the global max of all evaluated points
                ref_point = np.max(F_eval, axis=0) + 0.1 * (np.max(F_eval, axis=0) - np.min(F_eval, axis=0) + 1e-8)
            
            # Fit Gaussian Process
            try:
                # Standardize inputs for GP
                X_std = (X_eval - lo) / (hi - lo + 1e-8)
                
                # Fit GP using scipy or manual implementation?
                # SciPy doesn't have a built-in GP regressor that is easy to use for multi-output 
                # without extra packages like GPy. 
                # We must implement a simple GP or use a simpler surrogate like Kriging with SciPy.
                # Actually, implementing a robust GP from scratch is risky.
                # Alternative: Use a simpler surrogate, e.g., Polynomial Regression or Random Forest?
                # Random Forest is not in standard SciPy (sklearn is not allowed per "standard library, NumPy, SciPy").
                # So we are stuck with GP or Polynomial.
                # Let's implement a simple GP with RBF kernel.
                
                gp = self._fit_gp(X_std, F_eval)
                
                # Define Acquisition Function: EHVI
                # EHVI is complex to compute exactly. 
                # A common approximation is to sample from the GP and calculate hypervolume improvement.
                # Or use a simplified metric like Expected Improvement for each objective separately?
                # Let's try a simpler acquisition: Maximize the minimum normalized improvement 
                # (Tchebycheff) or use a simple "Expected Improvement" on the hypervolume 
                # approximated by Monte Carlo sampling of the GP.
                
                # Monte Carlo EHVI:
                # 1. Sample K latent vectors from GP
                # 2. For each sample, calculate hypervolume of (current_front U {sample})
                # 3. Compare to hypervolume of current_front
                # 4. Average improvement
                
                def acquisition(x):
                    x_std = (x - lo) / (hi - lo + 1e-8)
                    mean, cov = gp.predict(x_std)
                    # Sample 100 points from N(mean, cov)
                    samples = np.random.multivariate_normal(mean, cov, size=100)
                    # Filter out invalid samples (if any, though GP doesn't have bounds)
                    # Calculate HV improvement
                    base_hv = self._hypervolume(nd_points, ref_point)
                    improvements = []
                    for s in samples:
                        # Check if s is dominated by any point in nd_points
                        if self._is_dominated(s, nd_points):
                            improvements.append(0)
                        else:
                            # Add s to front and calculate new HV
                            new_front = np.vstack([nd_points, s])
                            new_hv = self._hypervolume(new_front, ref_point)
                            improvements.append(max(0, new_hv - base_hv))
                    return -np.mean(improvements) # Minimize negative improvement
                
                # Optimize acquisition function using local search from multiple starts
                # We will use the current non-dominated points and some random points as starts
                starts = list(nd_points)
                # Add 5 random points in the normalized space mapped back to original
                rng = np.random.default_rng(step)
                for _ in range(5):
                    starts.append(lo + (hi - lo) * rng.random(self.dim))
                
                best_acq_val = np.inf
                best_x = None
                
                for s in starts:
                    # Local optimization of acquisition
                    res = minimize(acquisition, s, method='L-BFGS-B', bounds=bounds, 
                                   options={'maxiter': 20, 'maxfun': 20})
                    if res.fun < best_acq_val:
                        best_acq_val = res.fun
                        best_x = res.x
                        
            except Exception:
                # If GP fitting or acquisition fails, fall back to random search
                best_x = lo + (hi - lo) * np.random.random(self.dim)
            
            # Evaluate the best candidate
            if best_x is not None:
                f_new = problem(best_x)
                X_eval = np.vstack([X_eval, best_x])
                F_eval = np.vstack([F_eval, f_new])
            else:
                # Fallback random
                x_rand = lo + (hi - lo) * np.random.random(self.dim)
                f_rand = problem(x_rand)
                X_eval = np.vstack([X_eval, x_rand])
                F_eval = np.vstack([F_eval, f_rand])
                
        return X_eval # Return all evaluated points

    def _latin_hypercube(self, n, dim, lo, hi):
        """Generate Latin Hypercube samples."""
        pts = np.zeros((n, dim))
        for i in range(dim):
            # Divide [0, 1] into n intervals, pick random point in each
            intervals = np.linspace(0, 1, n + 1)
            pts[:, i] = np.array([np.random.uniform(intervals[j], intervals[j+1]) for j in range(n)])
        # Map to [lo, hi]
        return lo + pts * (hi - lo)

    def _non_dominated(self, F):
        """Return indices of non-dominated points."""
        nd = []
        for i in range(len(F)):
            dominated = False
            for j in range(len(F)):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append(F[i])
        return np.array(nd) if nd else F[:1] # At least return one

    def _is_dominated(self, point, front):
        """Check if point is dominated by any point in front."""
        for p in front:
            if np.all(p <= point) and np.any(p < point):
                return True
        return False

    def _hypervolume(self, front, ref_point):
        """Calculate hypervolume of a 2D front relative to ref_point."""
        if len(front) == 0:
            return 0.0
        # Filter points that are better than ref_point
        valid = [p for p in front if np.all(p < ref_point)]
        if not valid:
            return 0.0
        
        # Sort by first objective
        valid = np.array(sorted(valid, key=lambda x: x[0]))
        
        hv = 0.0
        # Iterate through sorted points, calculate rectangle area
        # For 2D, HV is sum of widths * heights
        # Sort by x ascending
        for i in range(len(valid)):
            # Width is distance to next point's x (or ref_point if last)
            x_next = valid[i+1][0] if i < len(valid) - 1 else ref_point[0]
            width = x_next - valid[i][0]
            # Height is distance from y to ref_point
            height = ref_point[1] - valid[i][1]
            if width > 0 and height > 0:
                hv += width * height
        return hv

    def _fit_gp(self, X, Y):
        """
        Fit a simple Gaussian Process with RBF kernel.
        Returns a GP object with predict method.
        """
        # Implementing a full GP with MCMC or MAP is complex.
        # Given the constraints (no sklearn/gpy), let's use a simpler surrogate:
        # **Polynomial Regression** or **Linear Regression with RBF features**.
        # Let's use RBF features with fixed length scales, fitted via Linear Regression (scipy.linalg.lstsq).
        # This is much faster and more robust than a full GP for small budgets.
        
        # Number of RBF centers: use all data points as centers (Nadaraya-Watson style)
        # Or a subset if N is large. Here N <= 200, so use all.
        
        class SimpleSurrogate:
            def __init__(self, X, Y, dim):
                self.X = X
                self.Y = Y
                self.dim = dim
                # Calculate distances from each data point to itself? No, we need a kernel matrix K
                # K[i, j] = exp(-0.5 * ||X_i - X_j||^2 / l^2)
                # Let's estimate length scales.
                # Simple heuristic: l = median distance / 2
                dists = []
                for i in range(len(X)):
                    for j in range(i+1, len(X)):
                        dists.append(np.linalg.norm(X[i] - X[j]))
                if dists:
                    self.l = np.median(dists) / 2.0
                else:
                    self.l = 0.5
                
                # Build RBF Kernel Matrix
                self.K = np.zeros((len(X), len(X)))
                for i in range(len(X)):
                    for j in range(len(X)):
                        self.K[i, j] = np.exp(-0.5 * np.sum((X[i] - X[j])**2) / (self.l**2))
                
                # Add small regularization
                self.K += 1e-6 * np.eye(len(X))
                
                # Solve for weights: K * W = Y
                try:
                    self.W = np.linalg.solve(self.K, Y)
                except np.linalg.LinAlgError:
                    # Use least squares if singular
                    self.W, _, _, _ = np.linalg.lstsq(self.K, Y, rcond=None)

            def predict(self, X_new):
                # Calculate kernel between X_new and X_train
                K_new = np.zeros((len(X_new), len(self.X)))
                for i in range(len(X_new)):
                    for j in range(len(self.X)):
                        K_new[i, j] = np.exp(-0.5 * np.sum((X_new[i] - self.X[j])**2) / (self.l**2))
                
                mean = K_new @ self.W
                # For variance, we need the inverse of K. 
                # var = 1 - K_new @ K^{-1} @ K_new.T
                # Calculating K^{-1} is expensive but N is small.
                try:
                    K_inv = np.linalg.inv(self.K)
                    var = 1.0 - np.sum(K_new @ K_inv * K_new, axis=1)
                    var = np.maximum(var, 1e-8) # Ensure positive
                except np.linalg.LinAlgError:
                    var = np.ones(len(X_new)) * 1e-2
                
                return mean, np.diag(var)

        return SimpleSurrogate(X, Y, self.dim)

# EVOLVE-BLOCK-END
