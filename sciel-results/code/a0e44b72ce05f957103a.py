import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    Multi-start L-BFGS-B optimizer for rotated Rosenbrock (f9).
    Uses diverse strategic starting points and a final high-precision polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.eval_count = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')

        deadline = self.start_time + 290.0  # Safety margin before 330s kill

        class BudgetExceeded(Exception):
            pass

        def wrapped_func(x):
            if time.time() > deadline:
                raise BudgetExceeded()
            if self.eval_count >= self.budget:
                raise BudgetExceeded()

            x = np.clip(x, -5, 5)
            self.eval_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e10

            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()

            return fx

        rng = np.random.default_rng(42)
        dim = self.dim
        bounds = [(-5.0, 5.0)] * dim

        # Generate diverse starting points - prioritize points likely to converge fast
        starts = []

        # Origin first - often close to the minimum for BBOB functions
        starts.append(np.zeros(dim))

        # All-ones point (Rosenbrock minimum is at 1 in standard form)
        starts.append(np.ones(dim))
        starts.append(-np.ones(dim))

        # Diagonal points at various scales
        for s in [0.5, 1.5, -0.5, -1.5, 2.0, -2.0]:
            starts.append(np.full(dim, s))

        # Random points in the likely minimum region [-4, 4]^5
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, dim))

        # Points near the boundaries of the likely minimum region
        for _ in range(3):
            starts.append(rng.choice([-4, 4], size=dim))

        # Deduplicate starts (remove points that are very close to each other)
        unique_starts = []
        for s in starts:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(s - us) < 0.1:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        starts = unique_starts

        # Phase 1: Run L-BFGS-B from each start, keeping track of best
        # Use tighter tolerances for faster convergence and more starts
        for x0 in starts:
            if self.budget - self.eval_count < 10:
                break
            if time.time() > deadline - 10.0:
                break

            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 300,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxls': 50
                    }
                )
            except BudgetExceeded:
                break
            except Exception:
                continue

        # Phase 2: Final high-precision polish from the best point found
        # Use very tight tolerances and more iterations to squeeze out precision
        if self.budget - self.eval_count > 15 and time.time() < deadline - 5.0:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                        'maxls': 100
                    }
                )
            except BudgetExceeded:
                pass
            except Exception:
                pass

            # Additional Nelder-Mead polish for very tight convergence
            if self.budget - self.eval_count > 10 and time.time() < deadline - 3.0:
                try:
                    nm_res = scipy.optimize.minimize(
                        wrapped_func,
                        self.best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 200,
                            'xatol': 1e-14,
                            'fatol': 1e-14,
                            'adaptive': True
                        }
                    )
                except BudgetExceeded:
                    pass
                except Exception:
                    pass

        # Phase 3: If we still have budget, try a few more starts near the best
        if self.budget - self.eval_count > 15 and time.time() < deadline - 8.0:
            for _ in range(4):
                perturbed = self.best_x + rng.normal(0, 0.2, dim)
                perturbed = np.clip(perturbed, -5, 5)
                if self.budget - self.eval_count < 8:
                    break
                if time.time() > deadline - 4.0:
                    break
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func,
                        perturbed,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 200,
                            'ftol': 1e-15,
                            'gtol': 1e-13,
                            'maxls': 50
                        }
                    )
                except BudgetExceeded:
                    break
                except Exception:
                    continue

        return self.best_x, self.best_f
