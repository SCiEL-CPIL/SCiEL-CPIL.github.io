"""Bayesian Optimization for Multi-objective RE21 using EHVI acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Multi-objective Bayesian Optimizer using GP surrogates and EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize with random points
        n_init = min(20, self.budget)
        X = np.random.rand(n_init, self.dim) * (hi - lo) + lo
        F = np.array([problem(x) for x in X])
        
        # Filter out points worse than reference point [3000.0, 0.06]
        # But keep them in history for GP training, just don't count them for HV immediately
        # Actually, for EHVI, we need the current Pareto front.
        
        def get_pareto_front(F):
            mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not mask[j]:
                        continue
                    # If j dominates i, i is not on front
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return F[mask]
        
        # Initial Pareto front
        P = get_pareto_front(F)
        
        # GP Surrogates
        # We will fit a GP for each objective
        # Standardize inputs for better GP conditioning
        input_mean = X.mean(axis=0)
        input_std = X.std(axis=0) + 1e-8
        X_std = (X - input_mean) / input_std
        
        def fit_gp(X_std, y):
            """Fit a GP with RBF kernel and white noise."""
            n = len(y)
            # Normalize y for numerical stability
            y_mean = y.mean()
            y_std = y.std() + 1e-8
            y_norm = (y - y_mean) / y_std
            
            # Kernel parameters
            l = 0.5  # Length scale
            sigma_f = 1.0  # Signal variance
            sigma_n = 0.1  # Noise variance
            
            # Compute kernel matrix
            D = np.sum(X_std * X_std, axis=1)[:, None] + np.sum(X_std * X_std, axis=1)[None, :] - 2 * X_std @ X_std.T
            K = sigma_f**2 * np.exp(-0.5 * D / l**2) + sigma_n**2 * np.eye(n)
            
            # Solve for weights
            try:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
                v = np.linalg.solve(L, y_norm)
                w = np.linalg.solve(L.T, v)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if Cholesky fails
                w = np.linalg.pinv(K) @ y_norm
            
            return w, y_mean, y_std, l, sigma_f, sigma_n, input_mean, input_std

        def predict_gp(X_new, gp_params):
            """Predict mean and variance at new points."""
            w, y_mean, y_std, l, sigma_f, sigma_n, input_mean, input_std = gp_params
            X_new_std = (X_new - input_mean) / input_std
            X_train_std = (X - input_mean) / input_std
            
            D = np.sum(X_new_std * X_new_std, axis=1)[:, None] + np.sum(X_train_std * X_train_std, axis=1)[None, :] - 2 * X_new_std @ X_train_std.T
            K = sigma_f**2 * np.exp(-0.5 * D / l**2)
            
            mu = K @ w * y_std + y_mean
            var = sigma_f**2 - np.sum(K**2 * w[:, None] * w[None, :], axis=1)
            var = np.maximum(var, 1e-6)
            return mu, var

        # Fit initial GPs
        gps = []
        for j in range(self.n_obj):
            gps.append(fit_gp(X_std, F[:, j]))
        
        def ehvi_acquisition(X_cand, P, gps):
            """Calculate Expected Hypervolume Improvement for candidate points."""
            # Reference point
            ref = np.array([3000.0, 0.06])
            
            # Current hypervolume
            # For 2D, HV is sum of (ref[0]-p[0])*(ref[1]-p[1]) for each point, careful with overlaps
            # A robust way for small P is to sort by one objective and compute trapezoids
            
            def compute_hv(points, ref):
                # Filter points better than ref
                pts = points[np.all(points < ref, axis=1)]
                if len(pts) == 0:
                    return 0.0
                # Sort by first objective
                idx = np.argsort(pts[:, 0])
                pts = pts[idx]
                hv = 0.0
                prev_x = ref[0]
                for p in pts:
                    if p[0] >= prev_x:
                        break
                    # Width is prev_x - p[0]
                    # Height is min(prev_height, p[1]) - ref[1] ? No.
                    # In 2D, if sorted by x (obj1), the contribution of point i is (x_{i-1} - x_i) * (y_{i} - ref[1])
                    # Wait, this is only true if y is monotonic or we take the min y so far.
                    # Let's use the standard "layers" approach.
                    pass
                
                # Simpler 2D HV calculation:
                # Sort by obj1.
                # For each point, calculate the area of the rectangle it "owns".
                # A point owns the area to its right (higher obj1) and below (higher obj2)
                # up to the next point in obj1 or the ref point.
                
                # Let's just use a grid or the specific 2D formula.
                # Formula: Sort by obj1 ascending.
                # HV = sum_{i} (x_{i+1} - x_i) * (y_i - ref[1]) where x_{n+1}=ref[0], y_i is the min y seen so far?
                # No, for Pareto front, y is decreasing as x increases.
                # So HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * (y_i - ref[1]) + (ref[0] - x_n) * (y_n - ref[1])
                # This assumes the front is a step function. Which it is for a set of points.
                
                x = pts[:, 0]
                y = pts[:, 1]
                x_next = np.append(x[1:], ref[0])
                widths = x_next - x
                heights = y - ref[1]
                return np.sum(widths * heights)

            hv_current = compute_hv(P, ref)
            
            # Monte Carlo estimate of EHVI
            # Sample from GP posterior
            n_samples = 50
            best_ehvi = 0
            best_idx = 0
            
            # Pre-compute GP predictions for all candidates to save time?
            # Or loop candidates?
            # With budget 200 and dim 4, we can afford a decent number of candidates.
            # Let's use a local search or a set of random candidates + neighbors.
            
            # For each candidate, estimate EHVI
            # To speed up, we'll use a reduced set of candidates per iteration
            # 1. Current best points
            # 2. Random points
            # 3. Points near current best
            
            n_cands = 200
            # Generate candidates
            cand = np.random.rand(n_cands, self.dim) * (hi - lo) + lo
            # Add some jitter to current Pareto points
            if len(P) > 0:
                # We don't have X for P directly, but we can just rely on random search + exploitation
                # Actually, we should keep X corresponding to F.
                pass
            
            # We need X for P to generate local candidates.
            # Let's keep track of X_pareto.
            # Refactoring: Keep X_all, F_all.
            
            # This function needs access to X_all to generate local candidates.
            # Let's move candidate generation outside.
            
            # Predictions for all candidates
            mus = []
            vars_ = []
            for j in range(self.n_obj):
                mu, var = predict_gp(cand, gps[j])
                mus.append(mu)
                vars_.append(var)
            
            # For each candidate, sample
            ehvi_vals = np.zeros(n_cands)
            for i in range(n_cands):
                # Sample objective values
                y_samples = np.zeros((n_samples, self.n_obj))
                for j in range(self.n_obj):
                    y_samples[:, j] = mus[j][i] + np.sqrt(vars_[j][i]) * np.random.randn(n_samples)
                
                # For each sample, compute HV of (P union {y_sample})
                # And subtract HV(P)
                # Note: y_sample might be dominated by P, in which case HV doesn't change.
                
                # Optimization: Only compute HV if y_sample is non-dominated w.r.t P
                # Check dominance
                is_dom = False
                for p in P:
                    if np.all(p <= y_samples[0]) and np.any(p < y_samples[0]):
                        is_dom = True
                        break
                if is_dom:
                    # If dominated by at least one point, it might still contribute if it's not dominated by ALL?
                    # No, if it's dominated by ANY point in the Pareto front, it's not on the front.
                    # So HV doesn't change.
                    ehvi_vals[i] = 0
                    continue
                
                # It's not dominated by any single point? 
                # Wait, P is the Pareto front. If y is not dominated by any p in P, it might still be dominated by a combination?
                # No, dominance is pairwise. If no p in P dominates y, y is a candidate for the front.
                
                # Compute HV for all samples
                new_hvs = np.zeros(n_samples)
                for k in range(n_samples):
                    new_P = np.vstack([P, y_samples[k]])
                    new_P = get_pareto_front(new_P)
                    new_hvs[k] = compute_hv(new_P, ref)
                
                ehvi_vals[i] = np.mean(np.maximum(new_hvs - hv_current, 0))
            
            best_idx = np.argmax(ehvi_vals)
            return cand[best_idx]

        # Main Loop
        X_all = X
        F_all = F
        
        while len(X_all) < self.budget:
            if time.time() > deadline - 10:
                break
            
            # Update GPs with new data?
            # Yes, refit GPs periodically or every step?
            # Refitting every step is expensive. Let's refit every 5 steps or if we have time.
            # For now, refit every step if dim is small (4).
            X_std_all = (X_all - input_mean) / input_std
            gps = []
            for j in range(self.n_obj):
                gps.append(fit_gp(X_std_all, F_all[:, j]))
            
            # Update Pareto Front
            P = get_pareto_front(F_all)
            
            # Acquire next point
            x_next = ehvi_acquisition(X_all, P, gps)
            
            # Evaluate
            f_next = problem(x_next)
            
            # Add to history
            X_all = np.vstack([X_all, x_next])
            F_all = np.vstack([F_all, f_next])
            
            # Update input mean/std?
            # Maybe keep initial standardization to avoid drift?
            # Or update it. Let's update it to keep GP well-conditioned.
            input_mean = X_all.mean(axis=0)
            input_std = X_all.std(axis=0) + 1e-8
            
        return X_all
# EVOLVE-BLOCK-END
