"""Portfolio optimizer for BBOB f19 (Griewank-Rosenbrock) using GP surrogate + BFGS."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Multi-start portfolio optimizer with adaptive strategies."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = float('inf')
        observed_X = []
        observed_f = []

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            fx = float(fx)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            observed_X.append(x.copy())
            observed_f.append(fx)
            return x, fx

        def bfgs_local_search(x0, max_iter=50, tol=1e-8):
            """BFGS local search using the actual function."""
            nonlocal calls_made, best_x, best_f, observed_X, observed_f

            def objective(x):
                if calls_made >= budget or time.time() >= deadline:
                    return best_f
                x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                fx = float(fx)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                observed_X.append(x.copy())
                observed_f.append(fx)
                return fx

            try:
                res = minimize(objective, x0, method='BFGS', 
                              options={'maxiter': max_iter, 'gtol': tol, 'disp': False})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass
            return best_x, best_f

        def gp_predict(X_new, X_obs, f_obs, length_scale=1.0, noise=1e-6):
            """Gaussian Process prediction with RBF kernel."""
            X_obs = np.asarray(X_obs)
            f_obs = np.asarray(f_obs)
            X_new = np.asarray(X_new)

            # Standardize inputs
            mu = X_obs.mean(axis=0)
            sigma = X_obs.std(axis=0) + 1e-10
            X_obs_s = (X_obs - mu) / sigma
            X_new_s = (X_new - mu) / sigma

            # RBF kernel
            def rbf_kernel(A, B, ls):
                A = A / ls
                B = B / ls
                sq_dist = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
                sq_dist = np.maximum(sq_dist, 0)
                return np.exp(-0.5 * sq_dist)

            K = rbf_kernel(X_obs_s, X_obs_s, 1.0) + noise * np.eye(len(X_obs))
            Ks = rbf_kernel(X_obs_s, X_new_s, 1.0)

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_obs))
                v = np.linalg.solve(L, Ks.T)
                mu_pred = v.T @ alpha
                sigma2_pred = 1.0 - np.sum(v**2, axis=0)
                sigma_pred = np.sqrt(np.maximum(sigma2_pred, 1e-12))
            except np.linalg.LinAlgError:
                # Fallback to mean prediction
                mu_pred = np.full(len(X_new), np.mean(f_obs))
                sigma_pred = np.full(len(X_new), np.std(f_obs) + 0.1)

            return mu_pred, sigma_pred

        def expected_improvement(mu, sigma, best_f_val, xi=0.01):
            """Compute Expected Improvement."""
            improvement = best_f_val - mu - xi
            z = improvement / (sigma + 1e-10)
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0)

        def ei_acquisition(X_new, X_obs, f_obs, best_f_val, length_scale=1.0, noise=1e-6):
            """Expected Improvement acquisition function."""
            mu, sigma = gp_predict(X_new, X_obs, f_obs, length_scale, noise)
            ei = expected_improvement(mu, sigma, best_f_val)
            return ei

        # Phase 1: Initial diverse sampling
        rng = np.random.default_rng(42)
        n_init = min(15, budget // 3)
        
        # Include center and diverse points
        start_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            start_points.append(rng.uniform(-4, 4, size=dim))
        
        for sp in start_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            _, _ = safe_call(sp)

        if best_x is None:
            best_x = np.zeros(dim)

        # Phase 2: BFGS local search from best point
        if calls_made < budget and time.time() < deadline:
            bfgs_local_search(best_x, max_iter=30)

        # Phase 3: GP-guided exploration with EI acquisition
        min_obs = 10
        while calls_made < budget and time.time() < deadline:
            if len(observed_X) < min_obs:
                # Keep sampling randomly until we have enough data
                x_new = rng.uniform(-4, 4, size=dim)
                _, _ = safe_call(x_new)
                continue

            X_obs = np.array(observed_X)
            f_obs = np.array(observed_f)
            
            # Candidate points for EI maximization
            n_candidates = 500
            X_cand = rng.uniform(-4.5, 4.5, size=(n_candidates, dim))
            
            # Add some points near current best
            for _ in range(50):
                X_cand[len(X_cand)//10] = best_x + rng.normal(0, 0.5, size=dim)
                X_cand[len(X_cand)//10] = np.clip(X_cand[len(X_cand)//10], -4.5, 4.5)
            
            # Compute EI for all candidates
            ei_vals = ei_acquisition(X_cand, X_obs, f_obs, best_f)
            
            # Select top candidates
            top_k = min(10, len(ei_vals))
            top_idx = np.argsort(ei_vals)[-top_k:][::-1]
            
            # Evaluate the best EI point
            x_new = X_cand[top_idx[0]]
            x_new = np.clip(x_new, -5.0, 5.0)
            _, f_new = safe_call(x_new)
            
            # If we found improvement, do local search
            if f_new < best_f:
                if calls_made < budget and time.time() < deadline:
                    bfgs_local_search(best_x, max_iter=20)
            
            # Check for stagnation - if no improvement in last 20 calls, try different strategy
            if len(observed_f) >= 20:
                recent_best = min(observed_f[-20:])
                if recent_best >= best_f * 0.999 and calls_made < budget:
                    # Try a random restart with local search
                    x_restart = rng.uniform(-4, 4, size=dim)
                    _, _ = safe_call(x_restart)
                    if calls_made < budget and time.time() < deadline:
                        bfgs_local_search(x_restart, max_iter=15)

        # Final BFGS polish
        if calls_made < budget and time.time() < deadline and best_x is not None:
            bfgs_local_search(best_x, max_iter=30)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
