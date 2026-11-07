"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')

class Optimizer:
    """Surrogate-assisted Bayesian optimization with local polish for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            fx = float(func(x))
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # GP surrogate model
        class GPSurrogate:
            def __init__(self, dim, length_scale=1.0, noise=1e-4):
                self.dim = dim
                self.length_scale = length_scale
                self.noise = noise
                self.X = []
                self.y = []
                self.K_inv = None
                self.alpha = None
                self.y_mean = 0
                self.y_std = 1
                self.n = 0

            def _kernel(self, X1, X2):
                # Matern 5/2 kernel for smoothness
                d = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))
                return np.exp(-d / self.length_scale) * (1 + d / self.length_scale + d**2 / (3 * self.length_scale**2))

            def fit(self, X, y):
                X = np.asarray(X)
                y = np.asarray(y)
                self.X = X
                self.y = y
                self.n = len(y)
                self.y_mean = np.mean(y)
                self.y_std = max(np.std(y), 1e-6)
                y_norm = (y - self.y_mean) / self.y_std
                
                K = self._kernel(X, X) + self.noise * np.eye(self.n)
                try:
                    L = np.linalg.cholesky(K)
                    self.K_inv = np.linalg.inv(K)
                    self.alpha = np.linalg.solve(K, y_norm)
                except np.linalg.LinAlgError:
                    K_reg = K + 1e-6 * np.eye(self.n)
                    L = np.linalg.cholesky(K_reg)
                    self.K_inv = np.linalg.inv(K_reg)
                    self.alpha = np.linalg.solve(K_reg, y_norm)

            def predict(self, X_new):
                X_new = np.asarray(X_new)
                if len(self.X) == 0:
                    return np.zeros(len(X_new)), np.ones(len(X_new))
                
                K_s = self._kernel(X_new, self.X)
                mu = K_s @ self.alpha * self.y_std + self.y_mean
                K_ss = self._kernel(X_new, X_new)
                var = np.diag(K_ss) - np.sum(K_s @ (K_s.T @ self.K_inv), axis=1)
                var = np.maximum(var, 1e-12)
                return mu, np.sqrt(var)

            def expected_improvement(self, X_new, xi=0.01):
                mu, sigma = self.predict(X_new)
                best_y = np.min(self.y)
                improvement = best_y - mu - xi
                z = improvement / (sigma + 1e-12)
                from scipy.stats import norm
                EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                EI[improvement < 0] = 0
                return EI

        # Initialize GP
        gp = GPSurrogate(dim, length_scale=1.5, noise=1e-3)

        # Phase 1: Initial diverse sampling to build GP model
        n_initial = 40
        try:
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n=n_initial)
            samples = samples * 10.0 - 5.0
            initial_points = list(samples)
        except Exception:
            initial_points = np.random.uniform(-5, 5, size=(n_initial, dim)).tolist()

        # Add strategic points
        initial_points.append(np.zeros(dim))
        initial_points.append(np.ones(dim))
        initial_points.append(-np.ones(dim))
        initial_points.append(2.0 * np.ones(dim))
        initial_points.append(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))

        X_obs = []
        y_obs = []
        
        for p in initial_points:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                fp = wrapped_func(p)
                X_obs.append(np.asarray(p))
                y_obs.append(fp)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
                eval_count += 1
                X_obs.append(best_x.copy())
                y_obs.append(best_f)
            except Exception:
                pass

        # Fit initial GP
        if len(X_obs) >= 5:
            try:
                gp.fit(np.array(X_obs), np.array(y_obs))
            except Exception:
                pass

        # Phase 2: GP-guided Bayesian optimization loop
        n_bo_iter = 30
        for bo_iter in range(n_bo_iter):
            if time.time() > deadline or eval_count >= budget - 50:
                break
            
            # Generate candidate points for EI maximization
            n_candidates = 200
            try:
                sampler_cand = qmc.Sobol(d=dim, scramble=True)
                candidates = sampler_cand.random(n=n_candidates)
                candidates = candidates * 10.0 - 5.0
            except Exception:
                candidates = np.random.uniform(-5, 5, size=(n_candidates, dim))
            
            # Also add points near current best
            if best_x is not None:
                local_cands = best_x + np.random.randn(50, dim) * 0.5
                local_cands = np.clip(local_cands, -5, 5)
                candidates = np.vstack([candidates[:150], local_cands])
            
            # Compute EI for all candidates
            if len(X_obs) >= 5:
                try:
                    ei_vals = gp.expected_improvement(candidates, xi=0.05)
                    best_idx = np.argmax(ei_vals)
                except Exception:
                    best_idx = np.random.randint(n_candidates)
            else:
                best_idx = np.random.randint(n_candidates)
            
            x_new = candidates[best_idx]
            
            try:
                f_new = wrapped_func(x_new)
                X_obs.append(x_new.copy())
                y_obs.append(f_new)
                
                # Update GP
                if len(X_obs) >= 5:
                    try:
                        gp.fit(np.array(X_obs), np.array(y_obs))
                    except Exception:
                        pass
            except Exception:
                break

        # Phase 3: Local polish with L-BFGS-B from best point
        remaining = budget - eval_count
        if remaining > 30 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': min(remaining, 100), 'ftol': 1e-14, 'gtol': 1e-11}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: Additional GP-guided refinement if budget allows
        remaining = budget - eval_count
        if remaining > 20:
            n_refine = min(15, remaining // 3)
            for _ in range(n_refine):
                if time.time() > deadline or eval_count >= budget - 10:
                    break
                
                # Generate candidates focused on promising region
                n_cand = 100
                if best_x is not None:
                    cands = best_x + np.random.randn(n_cand, dim) * 0.3
                    cands = np.clip(cands, -5, 5)
                else:
                    cands = np.random.uniform(-3, 3, size=(n_cand, dim))
                
                if len(X_obs) >= 5:
                    try:
                        ei_vals = gp.expected_improvement(cands, xi=0.02)
                        best_idx = np.argmax(ei_vals)
                    except Exception:
                        best_idx = np.random.randint(n_cand)
                else:
                    best_idx = np.random.randint(n_cand)
                
                x_new = cands[best_idx]
                try:
                    f_new = wrapped_func(x_new)
                    X_obs.append(x_new.copy())
                    y_obs.append(f_new)
                    if len(X_obs) >= 5:
                        try:
                            gp.fit(np.array(X_obs), np.array(y_obs))
                        except Exception:
                            pass
                except Exception:
                    break

        # Phase 5: Final Nelder-Mead polish
        remaining = budget - eval_count
        if remaining > 15 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': min(remaining, 50), 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 6: Final L-BFGS-B pass
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': min(remaining, 30), 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
