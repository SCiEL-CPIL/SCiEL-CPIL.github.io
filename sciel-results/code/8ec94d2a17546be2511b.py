import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
import warnings

warnings.filterwarnings('ignore')

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s.
        # We aim for 300s limit in task, but use 290s as hard deadline to be safe.
        deadline = t0 + 290.0

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf
        self._best_x = None
        self._best_f = np.inf

        # Storage for Bayesian Optimization
        X_obs = []
        Y_obs = []

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 5:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            x_arr = np.asarray(x, dtype=float).flatten()
            val = float(func(x_arr))
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x_arr, dtype=float)
                self._best_x = best_x.copy()
                self._best_f = best_f
            return val

        # Phase 1: Initial Exploration (Latin Hypercube Sampling)
        # Use ~20% of budget for initial spread
        n_init = max(10, int(self.budget * 0.1))
        
        # Generate LHS points in [-5, 5]^dim
        n = self.dim
        lhs_points = np.zeros((n_init, n))
        for i in range(n_init):
            lhs_points[i] = rng.uniform(-5, 5, size=n)
        
        # Evaluate initial points
        for i in range(n_init):
            try:
                val = safe_func(lhs_points[i])
                X_obs.append(lhs_points[i])
                Y_obs.append(val)
            except ValueError:
                break

        if not X_obs:
            # Fallback if something went wrong immediately
            x0 = np.zeros(n)
            try:
                val = safe_func(x0)
                X_obs.append(x0)
                Y_obs.append(val)
            except:
                pass
        
        if not X_obs:
            return np.zeros(n), np.inf

        X_obs = np.array(X_obs)
        Y_obs = np.array(Y_obs)
        
        # Phase 2: Bayesian Optimization using RBF Interpolation
        # Use ~60% of budget for BO
        n_bo = int(self.budget * 0.6) - call_count
        if n_bo < 10:
            n_bo = 10
            
        # We will perform BO for a fixed number of iterations or until budget/time runs out
        # To keep it simple and robust, we'll loop until call_count hits the target or time is up.
        
        # RBF Interpolation parameters
        # 'thin_plate_spline' is often robust for optimization surfaces
        # 'multiquadric' is also good. Let's try 'thin_plate_spline' first.
        # Note: RBF can be unstable if points are too close or dimension is high.
        # For dim=5, it should be fine.
        
        # We need to handle the case where Y_obs has duplicates or issues.
        # RBFInterpolator requires unique points.
        
        # Instead of re-fitting RBF from scratch every time (which is O(N^3)),
        # we can just re-fit. N is small (max ~500), so 500x500 matrix is trivial.
        
        # Expected Improvement (EI) acquisition function
        # EI(x) = max(y_best - y_pred(x), 0) * Phi(z) + sigma(x) * phi(z)
        # where z = (y_best - y_pred(x)) / sigma(x)
        # Since RBF gives exact interpolation (sigma=0 at observed points), we need to add a small noise
        # or use a slightly different model to avoid division by zero in EI.
        # Alternatively, we can use a simple heuristic: Minimize the predicted value + uncertainty penalty?
        # Standard BO with GP is better, but GP is O(N^3) and might be slow in pure python/numpy without shogun/boTorch.
        # Let's stick to RBF + a simple exploration strategy.
        
        # Strategy:
        # 1. Fit RBF to current (X_obs, Y_obs).
        # 2. Generate a candidate grid or random points.
        # 3. Evaluate RBF prediction on candidates.
        # 4. Select candidate with lowest predicted value, but with some randomization to avoid getting stuck.
        #    Or use "Expected Improvement" approximation.
        
        # Let's try a simpler approach:
        # Use RBF to predict f(x).
        # To encourage exploration, we can add a small random jitter to the prediction or use a multi-start local search on the RBF surface.
        # Actually, a better approach for RBF is to find the minimum of the RBF surface itself.
        # But RBF surface might have local minima that are not real minima.
        
        # Let's use a hybrid:
        # 1. Fit RBF.
        # 2. Find the minimum of the RBF surface using L-BFGS-B from multiple random starts.
        # 3. Evaluate the best candidate found in the real function.
        
        # This is essentially "Surrogate-assisted local search".
        
        def fit_rbf():
            nonlocal X_obs, Y_obs
            if len(X_obs) < n + 1:
                return None
            try:
                # Ensure unique points
                unique_idx = np.unique(np.arange(len(X_obs)), axis=0)
                if len(unique_idx) < len(X_obs):
                    # This shouldn't happen with random sampling, but just in case
                    pass
                
                # RBFInterpolator can be sensitive to scale. Normalize X?
                # X is in [-5, 5]. That's fine.
                
                rbf = RBFInterpolator(X_obs, Y_obs, kernel='thin_plate_spline', smoothing=0.1)
                return rbf
            except Exception:
                # If RBF fails, try 'multiquadric'
                try:
                    rbf = RBFInterpolator(X_obs, Y_obs, kernel='multiquadric', smoothing=0.1)
                    return rbf
                except Exception:
                    return None

        # Loop for BO
        for _ in range(n_bo):
            if call_count >= self.budget or time.time() > deadline - 20:
                break
                
            rbf = fit_rbf()
            
            if rbf is None:
                # Fallback to random sampling
                x_new = rng.uniform(-5, 5, size=n)
            else:
                # Find minimum of RBF surface
                # Use multiple starts to avoid local minima of the surrogate
                best_rbf_x = None
                best_rbf_val = np.inf
                
                n_starts = 5
                for start_i in range(n_starts):
                    if start_i == 0:
                        x_start = X_obs[np.argmin(Y_obs)]
                    else:
                        x_start = rng.uniform(-5, 5, size=n)
                    
                    try:
                        res = minimize(
                            lambda x: rbf(x.reshape(1, -1))[0],
                            x_start,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * n,
                            options={'maxiter': 100, 'ftol': 1e-10}
                        )
                        if res.fun < best_rbf_val:
                            best_rbf_val = res.fun
                            best_rbf_x = res.x
                    except Exception:
                        continue
                
                if best_rbf_x is not None:
                    x_new = best_rbf_x
                else:
                    x_new = rng.uniform(-5, 5, size=n)
            
            # Evaluate the candidate
            try:
                val = safe_func(x_new)
                X_obs.append(x_new)
                Y_obs.append(val)
            except ValueError:
                break

        # Phase 3: Local Refinement with L-BFGS-B
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 20:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 1000, 'maxfun': remaining, 'maxls': 50, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 4: Nelder-Mead Polish
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 10:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 5: Final high-precision L-BFGS-B polish
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            try:
                res_final = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 1000, 'maxfun': remaining, 'maxls': 100, 
                             'gtol': 1e-15, 'ftol': 1e-16, 'disp': False}
                )
                if res_final.fun < best_f:
                    best_f = float(res_final.fun)
                    best_x = np.array(res_final.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        return best_x, best_f
