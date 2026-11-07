"""Hybrid GP-Bayesian Optimization + Multi-start BFGS Local Search for BBOB f19."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
import math


class Optimizer:
    """
    Hybrid optimizer combining Gaussian Process Bayesian Optimization (global)
    with multi-start BFGS local search (local) for BBOB f19 (Griewank-Rosenbrock).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _matern_52_kernel(self, A, B, length_scale):
        """Compute Matern 5/2 kernel between sets of points A and B."""
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=-1))
        sr = np.sqrt(5.0) * r / length_scale
        K = (1.0 + sr + sr * sr / 3.0) * np.exp(-sr)
        return K

    def _gp_predict(self, X, X_obs, y_obs, length_scale, noise):
        """
        Predict mean and variance at X using GP.
        """
        n = X_obs.shape[0]
        if n == 0:
            return np.zeros(X.shape[0]), np.ones(X.shape[0])
        
        K = self._matern_52_kernel(X_obs, X_obs, length_scale)
        K[np.diag_indices_from(K)] += noise
        
        KX = self._matern_52_kernel(X_obs, X, length_scale)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            v = np.linalg.solve(L, KX)
        except np.linalg.LinAlgError:
            K[np.diag_indices_from(K)] += 1e-6
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
                v = np.linalg.solve(L, KX)
            except np.linalg.LinAlgError:
                return np.zeros(X.shape[0]), np.ones(X.shape[0])

        KXX = self._matern_52_kernel(X, X, length_scale)
        mean = KX.T @ alpha
        var = np.diag(KXX) - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return mean, var

    def _expected_improvement(self, mu, sigma, f_best, xi=0.01):
        """Compute Expected Improvement acquisition function."""
        sigma = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei

    def _run_bfgs(self, func, x0, calls_made, best_f, best_x, budget, deadline, n):
        """Run BFGS from x0, returning (new_calls, new_best_f, new_best_x)."""
        local_best_f = best_f
        local_best_x = best_x.copy()
        
        def bfgs_func(x):
            nonlocal calls_made, local_best_f, local_best_x
            if calls_made >= budget or time.time() > deadline - 3:
                return local_best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = func(x)
            calls_made += 1
            if val < local_best_f:
                local_best_f = val
                local_best_x = x.copy()
            return val

        try:
            res = minimize(bfgs_func, x0, method='BFGS', 
                           options={'maxiter': 50, 'gtol': 1e-7})
            if res.fun < local_best_f:
                local_best_f = res.fun
                local_best_x = np.clip(res.x, -5.0, 5.0).copy()
        except Exception:
            pass
        
        return calls_made, local_best_f, local_best_x

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        n = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        # State for GP
        X_obs = []
        y_obs = []
        
        # Adaptive parameters
        length_scale = 2.0
        noise = 1e-6

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget or time.time() > deadline - 2:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = func(x)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # --- Phase 1: Initial Exploration ---
        init_points = [np.zeros(n)]
        for _ in range(9):
            init_points.append(rng.uniform(-4.0, 4.0, n))
        
        for p in init_points:
            if calls_made >= self.budget or time.time() > deadline - 20:
                break
            val = safe_call(p)
            if val is not None:
                X_obs.append(p)
                y_obs.append(val)

        # --- Phase 2: GP-Guided Search with Multi-start BFGS Exploitation ---
        batch_size = 4
        last_bfgs_call = 0
        stagnant_calls = 0
        prev_best_f = best_f

        while calls_made < self.budget and time.time() < deadline - 15:
            if len(X_obs) < 2:
                p = rng.uniform(-4.0, 4.0, n)
                val = safe_call(p)
                if val is not None:
                    X_obs.append(p)
                    y_obs.append(val)
                continue

            X_obs_arr = np.array(X_obs)
            y_obs_arr = np.array(y_obs)
            f_best_current = np.min(y_obs_arr)

            # Generate candidate points
            n_candidates = 120
            candidates = rng.uniform(-4.5, 4.5, (n_candidates, n))
            
            if best_x is not None:
                local_candidates = best_x + rng.normal(0, 0.4, (40, n))
                local_candidates = np.clip(local_candidates, -4.5, 4.5)
                candidates = np.vstack([candidates, local_candidates])

            # Predict GP
            mu, var = self._gp_predict(candidates, X_obs_arr, y_obs_arr, length_scale, noise)
            sigma = np.sqrt(var)
            
            # Compute EI
            ei = self._expected_improvement(mu, sigma, f_best_current, xi=0.005)
            
            # Select top batch_size candidates with diversity
            ei_sorted_idx = np.argsort(-ei)
            selected = []
            for idx in ei_sorted_idx:
                if len(selected) >= batch_size:
                    break
                cand = candidates[idx]
                if selected:
                    dists = np.array([np.linalg.norm(cand - s) for s in selected])
                    if np.min(dists) < 0.15:
                        continue
                selected.append(cand)
            
            if len(selected) < batch_size:
                for idx in ei_sorted_idx:
                    if len(selected) >= batch_size:
                        break
                    cand = candidates[idx]
                    if cand not in selected:
                        selected.append(cand)

            # Evaluate selected points
            for cand in selected:
                if calls_made >= self.budget or time.time() > deadline - 15:
                    break
                val = safe_call(cand)
                if val is not None:
                    X_obs.append(cand)
                    y_obs.append(val)

            # Track stagnation
            if best_f < prev_best_f:
                stagnant_calls = 0
            else:
                stagnant_calls += 1
            prev_best_f = best_f

            # --- Multi-start BFGS Exploitation ---
            # Trigger conditions: periodic, or during stagnation
            should_run_bfgs = False
            if calls_made < self.budget - 30 and time.time() < deadline - 25:
                if (calls_made - last_bfgs_call >= 15) or (stagnant_calls >= 4 and len(X_obs) > 8):
                    should_run_bfgs = True

            if should_run_bfgs:
                last_bfgs_call = calls_made
                # Get top-K best points for multi-start
                if len(X_obs) >= 3:
                    y_arr = np.array(y_obs)
                    # Sort by objective value, take top 3 distinct points
                    sorted_idx = np.argsort(y_arr)
                    start_points = []
                    for idx in sorted_idx[:3]:
                        p = X_obs[idx]
                        if not any(np.linalg.norm(p - sp) < 0.3 for sp in start_points):
                            start_points.append(p)
                        if len(start_points) >= 3:
                            break
                else:
                    start_points = [best_x.copy()] if best_x is not None else []

                for sp in start_points:
                    if calls_made >= self.budget - 10 or time.time() > deadline - 10:
                        break
                    calls_made, bf, bx = self._run_bfgs(
                        func, sp, calls_made, best_f, best_x, 
                        self.budget, deadline, n
                    )
                    best_f = bf
                    best_x = bx
                    # Add the BFGS result to GP observations
                    if bx is not None and bf < float('inf'):
                        X_obs.append(bx.copy())
                        y_obs.append(bf)

            # Adapt length scale
            if best_f < 1.0:
                length_scale = max(0.3, length_scale * 0.93)
            elif best_f < 10.0:
                length_scale = max(0.5, length_scale * 0.96)

        # --- Phase 3: Final High-Precision Polish ---
        if calls_made < self.budget and time.time() < deadline - 2 and best_x is not None:
            # Final BFGS with tighter tolerance
            x0 = best_x.copy()
            
            def final_func(x):
                nonlocal calls_made, best_f, best_x
                if calls_made >= self.budget or time.time() > deadline - 1:
                    return best_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
                val = func(x)
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                res = minimize(final_func, x0, method='BFGS', 
                               options={'maxiter': 100, 'gtol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

            # (1+1)-ES polish with remaining budget
            if calls_made < self.budget and time.time() < deadline - 1:
                polish_x = best_x.copy()
                polish_f = best_f
                polish_sigma = 0.02
                
                while calls_made < self.budget and time.time() < deadline - 0.5:
                    step = polish_sigma * rng.standard_normal(n)
                    candidate = np.clip(polish_x + step, -5, 5)
                    candidate_f = func(candidate)
                    calls_made += 1
                    
                    if candidate_f < polish_f:
                        polish_x = candidate
                        polish_f = candidate_f
                        polish_sigma *= 1.15
                        if polish_f < best_f:
                            best_f = polish_f
                            best_x = polish_x.copy()
                    else:
                        polish_sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
