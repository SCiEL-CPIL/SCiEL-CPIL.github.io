"""Bayesian Optimization with Gaussian Process surrogate for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Bayesian Optimization using a Gaussian Process surrogate to model the
    multimodal Weierstrass landscape. Uses Expected Improvement acquisition
    for global exploration, combined with L-BFGS-B local refinement on the
    best candidates found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                val = float(func(x_clipped))
                return val, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                return True
            return False

        rng = np.random.default_rng(42)

        # --- Gaussian Process Surrogate ---
        class GPSurrogate:
            """Simple GP with Matern 5/2 kernel."""
            def __init__(self, dim, length_scale=1.0, signal_var=1.0, noise_var=1e-6):
                self.dim = dim
                self.length_scale = length_scale
                self.signal_var = signal_var
                self.noise_var = noise_var
                self.X = None
                self.Y = None
                self.K_inv = None
                self.alpha = None

            def _kernel(self, X1, X2):
                # Matern 5/2 kernel
                d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                d2 = np.maximum(d2, 0)
                d = np.sqrt(d2)
                sqrt3d = np.sqrt(3) * d / self.length_scale
                return self.signal_var * (1 + sqrt3d + sqrt3d**2 / 3) * np.exp(-sqrt3d)

            def fit(self, X, Y):
                self.X = np.asarray(X, dtype=float)
                self.Y = np.asarray(Y, dtype=float)
                n = len(X)
                K = self._kernel(X, X) + self.noise_var * np.eye(n)
                # Regularize for numerical stability
                K += 1e-8 * np.eye(n)
                try:
                    self.K_inv = np.linalg.inv(K)
                    self.alpha = self.K_inv @ self.Y
                except np.linalg.LinAlgError:
                    # Fallback: use diagonal approximation
                    self.K_inv = np.diag(1.0 / np.diag(K))
                    self.alpha = self.K_inv @ self.Y

            def predict(self, X_new):
                X_new = np.asarray(X_new, dtype=float)
                if X_new.ndim == 1:
                    X_new = X_new.reshape(1, -1)
                K_s = self._kernel(X_new, self.X)
                mean = K_s @ self.alpha
                K_ss = self._kernel(X_new, X_new)
                var = np.diag(K_ss) - np.sum(K_s**2 * self.alpha[None, :], axis=1)
                var = np.maximum(var, 1e-10)
                return mean, np.sqrt(var)

            def expected_improvement(self, X_new, f_best, xi=0.01):
                mean, std = self.predict(X_new)
                # For minimization: EI = (f_best - mean) * Phi(z) + std * phi(z)
                # where z = (f_best - mean) / std
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = (f_best - mean) / std
                    z = np.where(std > 1e-10, z, 0)
                    # Use standard normal CDF and PDF
                    from scipy.stats import norm
                    ei = (f_best - mean) * norm.cdf(z) + std * norm.pdf(z)
                    ei = np.where(std > 1e-10, ei, 0)
                return ei

        # --- L-BFGS-B Local Polisher ---
        def local_polish_l_bfgs(x0, f0, max_calls):
            """
            Run L-BFGS-B from x0. Returns best_x, best_f, calls_used.
            """
            nonlocal n_calls, best_x, best_f

            calls_used = 0
            x_best = x0.copy()
            f_best = f0

            def objective(x):
                nonlocal calls_used
                if calls_used >= max_calls or n_calls >= budget or check_time():
                    return f_best
                val, x_eval = call_func(x)
                if val is None:
                    return f_best
                calls_used += 1
                return val

            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max_calls, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < f_best:
                    x_best = np.clip(result.x, -5, 5)
                    f_best = result.fun
                    if f_best < best_f:
                        best_f = f_best
                        best_x = x_best.copy()
            except Exception:
                pass

            return x_best, f_best, calls_used

        # --- Phase 1: Initial Sampling ---
        # Sample 15 points using a quasi-random (Halton-like) or random strategy
        n_initial = 15
        X_obs = []
        Y_obs = []
        
        for i in range(n_initial):
            if n_calls >= budget or check_time():
                break
            x = rng.uniform(-5, 5, dim)
            f, x_eval = call_func(x)
            if f is None:
                continue
            X_obs.append(x_eval)
            Y_obs.append(f)
            update_best(x_eval, f)

        if best_x is None:
            return best_x, best_f

        # Fit initial GP
        gp = GPSurrogate(dim, length_scale=1.5, signal_var=np.var(Y_obs) + 1e-6 if len(Y_obs) > 1 else 1.0, noise_var=1e-4)
        gp.fit(np.array(X_obs), np.array(Y_obs))

        # --- Phase 2: Bayesian Optimization Loop ---
        n_bo_iter = 30  # Number of BO iterations
        n_acq_samples = 2000  # Number of samples for acquisition function optimization
        
        for bo_iter in range(n_bo_iter):
            if n_calls >= budget or check_time():
                break
            
            # Update GP with all observed points
            if len(X_obs) >= 2:
                gp.fit(np.array(X_obs), np.array(Y_obs))
            else:
                # Not enough data for GP, use random sampling
                x = rng.uniform(-5, 5, dim)
                f, x_eval = call_func(x)
                if f is None:
                    continue
                X_obs.append(x_eval)
                Y_obs.append(f)
                update_best(x_eval, f)
                continue

            # Find next point to evaluate using EI acquisition
            # Sample candidate points and pick the one with highest EI
            X_candidates = rng.uniform(-5, 5, (n_acq_samples, dim))
            ei = gp.expected_improvement(X_candidates, best_f, xi=0.01)
            
            # Add some exploration: mix EI with random points
            # Top 80% by EI, bottom 20% random
            top_ei_idx = np.argsort(ei)[-1600:]  # Top 80%
            random_idx = rng.choice(n_acq_samples, size=400, replace=False)
            candidate_idx = np.concatenate([top_ei_idx, random_idx])
            candidate_idx = rng.choice(candidate_idx, size=50, replace=False)
            
            # Evaluate top candidate
            x_next = X_candidates[candidate_idx[0]]
            f_next, x_next_eval = call_func(x_next)
            if f_next is None:
                continue
            
            X_obs.append(x_next_eval)
            Y_obs.append(f_next)
            update_best(x_next_eval, f_next)

            # Every 5 BO iterations, do local polish on the best point
            if (bo_iter + 1) % 5 == 0 and best_x is not None:
                x_polished, f_polished, calls_used = local_polish_l_bfgs(best_x, best_f, max_calls=10)
                if f_polished < best_f:
                    X_obs.append(x_polished)
                    Y_obs.append(f_polished)

        # --- Phase 3: Final Local Polish ---
        # Use remaining budget for final L-BFGS-B polish
        if best_x is not None and n_calls < budget and not check_time():
            remaining = budget - n_calls
            max_polish = min(remaining, 50)
            if max_polish > 5:
                x_polished, f_polished, calls_used = local_polish_l_bfgs(best_x, best_f, max_calls=max_polish)
                if f_polished < best_f:
                    best_x = x_polished
                    best_f = f_polished

        return best_x, best_f

# EVOLVE-BLOCK-END
