import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(seed=42)

        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(x, -self.bound, self.bound)
            fx = func(x)
            if np.isnan(fx) or np.isinf(fx):
                fx = np.inf
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        def safe_minimize(method, x0, **kwargs):
            """Run scipy minimize with safety guards."""
            try:
                res = scipy.optimize.minimize(wrapped_func, x0, method=method, **kwargs)
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 1: Global random sampling
        # 5D space, need good coverage. Use 80 random points.
        n_global = 80
        for _ in range(n_global):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = rng.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)

        if self.best_x is None:
            # Should not happen, but fallback
            return np.zeros(self.dim), 0.0

        # Phase 2: Multi-start local optimization
        # Use the best point found plus a few nearby perturbations
        # Try Powell from the global best
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 20 and time.time() < self.deadline - 10:
            safe_minimize(
                'Powell',
                self.best_x,
                options={'maxiter': 30, 'ftol': 1e-12, 'xtol': 1e-12}
            )

        # Phase 3: Try Nelder-Mead from best
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 20 and time.time() < self.deadline - 10:
            safe_minimize(
                'Nelder-Mead',
                self.best_x,
                options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )

        # Phase 4: If budget remains, do a second round of global sampling
        # focused around the best point (local exploration)
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 30 and time.time() < self.deadline - 15:
            n_local_samples = min(30, remaining_budget // 2)
            for _ in range(n_local_samples):
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                # Sample around best point with Gaussian perturbation
                p = self.best_x + rng.normal(0, 0.5, self.dim)
                p = np.clip(p, -self.bound, self.bound)
                wrapped_func(p)

        # Phase 5: Final polish with Nelder-Mead if budget allows
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 15 and time.time() < self.deadline - 10:
            safe_minimize(
                'Nelder-Mead',
                self.best_x,
                options={'maxiter': 50, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )

        return self.best_x, self.best_f
