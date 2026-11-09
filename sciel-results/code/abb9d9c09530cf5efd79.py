import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    Multi-start L-BFGS-B optimizer for rotated Rosenbrock (f9)
    with Latin Hypercube sampling for diverse starts and a final polish pass.
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

        class BudgetExceeded(Exception):
            pass

        def wrapped_func(x):
            if time.time() - self.start_time > 285:
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
        starts = []

        # Priority 1: Origin (Rosenbrock's known minimum location)
        starts.append(np.zeros(self.dim))

        # Priority 2: Points along the valley direction (x=1, y=1, ...)
        # In standard Rosenbrock, the valley is near x_i ≈ 1
        valley_point = np.ones(self.dim)
        starts.append(valley_point)

        # Priority 3: Scaled valley points
        for scale in [0.5, 1.5, 2.0, -1.0]:
            starts.append(scale * np.ones(self.dim))

        # Priority 4: LHS points for coverage
        n_lhs = 15
        for i in range(n_lhs):
            point = np.zeros(self.dim)
            for d in range(self.dim):
                offset = rng.random()
                val = -4 + (i + offset) * (8.0 / n_lhs)
                point[d] = val
            starts.append(point)

        # Priority 5: Additional random points
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, self.dim))

        bounds = [(-5, 5)] * self.dim

        # Sort: origin first, then by distance from origin
        starts.sort(key=lambda x: np.linalg.norm(x))

        # Phase 1: Multi-start L-BFGS-B
        for x0 in starts:
            if self.budget - self.eval_count < 15:
                break
            if time.time() - self.start_time > 285:
                break

            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-10,
                        'gtol': 1e-6,
                        'maxls': 20
                    }
                )
            except BudgetExceeded:
                break
            except Exception:
                continue

        # Phase 2: High-precision polish from best point
        if self.eval_count < self.budget - 20 and time.time() - self.start_time < 280:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-15,
                        'gtol': 1e-10,
                        'maxls': 30
                    }
                )
            except BudgetExceeded:
                pass
            except Exception:
                pass

        # Phase 3: If budget remains, try CMA-ES from best point
        if self.eval_count < self.budget - 30 and time.time() - self.start_time < 275:
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead as a derivative-free polish
                res = minimize(
                    wrapped_func,
                    self.best_x.copy(),
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except BudgetExceeded:
                pass
            except Exception:
                pass

        # Final polish if still have budget
        if self.eval_count < self.budget - 5 and time.time() - self.start_time < 270:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-18,
                        'gtol': 1e-12,
                        'maxls': 20
                    }
                )
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return self.best_x, self.best_f
