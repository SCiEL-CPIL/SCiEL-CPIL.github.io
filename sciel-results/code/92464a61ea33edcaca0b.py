"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted (Kriging) global optimizer with local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def timed_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                raise RuntimeError("Budget exceeded")
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            call_count += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial sampling with Latin Hypercube
        n_init = 20
        X_init = np.zeros((n_init, n))
        for i in range(n_init):
            X_init[i] = rng.uniform(-5.0, 5.0, n)
        
        f_init = np.array([timed_func(X_init[i]) for i in range(n_init)])
        X_obs = X_init.copy()
        f_obs = f_init.copy()

        # Phase 2: Surrogate-assisted search with Kriging
        # Fit a simple Kriging model and use EI acquisition
        def fit_kriging(X, f):
            """Fit a simple Kriging model with constant mean + ARD kernel."""
            m = len(X)
            if m < 2:
                return lambda x: np.mean(f)
            
            # Normalize inputs to [0,1] for better conditioning
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range < 1e-10] = 1.0
            X_norm = (X - X_min) / X_range
            
            # Fit constant mean
            f_mean = np.mean(f)
            f_centered = f - f_mean
            
            # ARD kernel: k(x, x') = exp(-sum_j (x_j - x'_j)^2 / theta_j^2)
            # Use a simple isotropic kernel for stability
            theta2 = 1.0  # length-scale squared
            
            # Build kernel matrix
            K = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    diff = X_norm[i] - X_norm[j]
                    K[i, j] = np.exp(-np.sum(diff**2) / theta2)
            
            # Add small jitter for numerical stability
            K += 1e-8 * np.eye(m)
            
            # Solve for weights
            try:
                K_inv = np.linalg.inv(K)
                weights = K_inv @ f_centered
            except np.linalg.LinAlgError:
                weights = np.zeros(m)
            
            def predict(x):
                x = np.asarray(x, dtype=np.float64)
                x_norm = (x - X_min) / X_range
                # Mean prediction
                mu = f_mean + sum(weights[i] * np.exp(-np.sum((x_norm - X_norm[i])**2) / theta2) 
                                  for i in range(m))
                # Variance (simplified)
                var = 1.0
                return mu, var
            
            return predict

        def expected_improvement(x, mu, sigma, best_f, xi=0.01):
            """Expected Improvement acquisition function."""
            if sigma < 1e-10:
                return 0.0
            imp = best_f - mu - xi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return max(ei, 0.0)

        # EI maximization via multi-start local optimization
        def maximize_ei(predictor, best_f):
            if call_count >= budget - 10:
                return None
            
            # Evaluate EI at random points to find good starting points
            n_candidates = 50
            X_cand = rng.uniform(-5.0, 5.0, (n_candidates, n))
            ei_vals = np.zeros(n_candidates)
            for i in range(n_candidates):
                mu, sigma = predictor(X_cand[i])
                ei_vals[i] = expected_improvement(X_cand[i], mu, sigma, best_f)
            
            # Take top 5 candidates and refine with local optimization
            top_idx = np.argsort(ei_vals)[-5:]
            best_ei = -np.inf
            best_x_ei = None
            
            for idx in top_idx:
                x0 = X_cand[idx].copy()
                
                def neg_ei(x):
                    mu, sigma = predictor(x)
                    return -expected_improvement(x, mu, sigma, best_f)
                
                try:
                    res = minimize(neg_ei, x0, method='Nelder-Mead',
                                   options={'maxiter': 30, 'xatol': 0.01, 'fatol': 1e-6,
                                           'adaptive': True})
                    if res.fun < best_ei:
                        best_ei = res.fun
                        best_x_ei = res.x
                except Exception:
                    continue
            
            if best_x_ei is not None:
                return np.clip(best_x_ei, -5.0, 5.0)
            return None

        # Main surrogate loop
        n_surrogate_steps = min(200, budget - call_count - 50)
        for step in range(n_surrogate_steps):
            if call_count >= budget - 50 or time.time() > deadline:
                break
            
            # Fit surrogate
            predictor = fit_kriging(X_obs, f_obs)
            
            # Find next point via EI
            x_next = maximize_ei(predictor, best_f)
            if x_next is None:
                break
            
            # Evaluate
            f_next = timed_func(x_next)
            X_obs = np.vstack([X_obs, x_next])
            f_obs = np.append(f_obs, f_next)
            
            # Check for convergence
            if step > 50 and (best_f - f_obs[-1]) < 1e-10:
                break

        # Phase 3: Local polish with L-BFGS-B from best point
        if best_x is not None and call_count < budget - 10 and time.time() < deadline:
            start_points = [best_x.copy()]
            # Add a few perturbed points
            for i in range(3):
                perturbation = rng.standard_normal(n) * 0.3
                xp = best_x + perturbation
                xp = np.clip(xp, -5.0, 5.0)
                start_points.append(xp)
            
            for sp in start_points:
                if call_count >= budget - 5 or time.time() > deadline:
                    break
                try:
                    res = minimize(timed_func, sp, method='L-BFGS-B',
                                  bounds=[(-5.0, 5.0)] * n,
                                  options={'maxiter': 30, 'ftol': 1e-15, 'gtol': 1e-15})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        # Phase 4: Final Nelder-Mead polish
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            try:
                res = minimize(timed_func, best_x, method='Nelder-Mead',
                              options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12,
                                      'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Ensure valid result
        if best_x is None:
            best_x = np.zeros(n)
            best_f = timed_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
