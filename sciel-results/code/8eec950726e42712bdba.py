import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid BFGS + L-BFGS-B + Nelder-Mead Optimizer for Rosenbrock-like functions.
    
    Uses a derivative-free BFGS quasi-Newton method as the primary search engine,
    approximating the Hessian from function evaluations to navigate Rosenbrock's
    valley curvature more efficiently than finite-difference gradients.
    L-BFGS-B provides a fallback for robust convergence, and Nelder-Mead polishes
    to high precision. Multi-start with diverse initializations handles hidden
    shift/rotation/offset.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 285.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = float(func(x))
            if not np.isfinite(val):
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Priority-ordered diverse starting points
        starts = [np.ones(self.dim)]

        # Small perturbations around standard Rosenbrock minimum
        for scale in [0.3, 0.7, 1.5]:
            starts.append(np.ones(self.dim) + rng.normal(0, scale, self.dim))

        # Structured points: center, corners of likely region
        starts.append(np.zeros(self.dim))
        starts.append(np.full(self.dim, -1.0))
        starts.append(np.full(self.dim, 1.0))
        starts.append(np.full(self.dim, 2.0))
        starts.append(np.full(self.dim, -2.0))
        starts.append(np.full(self.dim, 3.0))

        # Random in likely region [-4, 4]
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, self.dim))

        # A few in full box for robustness
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Phase 1: BFGS quasi-Newton for fast valley navigation
        for x0 in unique_starts:
            if self.calls_made >= self.budget - 60:
                break
            if time.time() > self.deadline:
                break

            x0 = np.clip(x0, -5, 5)
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 30:
                break

            try:
                res_bfgs = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='BFGS',
                    options={
                        'maxiter': 150,
                        'gtol': 1e-5,
                        'ftol': 1e-8,
                        'maxfun': remaining_budget
                    }
                )

                if np.isfinite(res_bfgs.fun) and res_bfgs.fun < self.best_f:
                    self.best_f = res_bfgs.fun
                    self.best_x = np.clip(res_bfgs.x, -5, 5)

            except StopIteration:
                break
            except Exception:
                continue

        # Phase 2: L-BFGS-B fallback for robust convergence
        for x0 in unique_starts[:8]:
            if self.calls_made >= self.budget - 40:
                break
            if time.time() > self.deadline:
                break

            x0 = np.clip(x0, -5, 5)
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 20:
                break

            try:
                res_lbfgsb = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-10,
                        'gtol': 1e-6,
                        'maxfun': remaining_budget
                    }
                )

                if np.isfinite(res_lbfgsb.fun) and res_lbfgsb.fun < self.best_f:
                    self.best_f = res_lbfgsb.fun
                    self.best_x = np.clip(res_lbfgsb.x, -5, 5)

            except StopIteration:
                break
            except Exception:
                continue

        # Phase 3: Nelder-Mead polish from the best point found
        if self.calls_made < self.budget - 10 and self.best_x is not None:
            try:
                x_start = np.clip(self.best_x.copy(), -5, 5)

                res_nm = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-9,
                        'fatol': 1e-9,
                        'maxiter': 100,
                        'adaptive': True
                    }
                )

                if np.isfinite(res_nm.fun) and res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = np.clip(res_nm.x, -5, 5)

            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 4: Final high-precision polish if budget allows
        if self.calls_made < self.budget - 2 and self.best_x is not None and self.best_f < 1e4:
            try:
                x_start = np.clip(self.best_x.copy(), -5, 5)

                res_nm2 = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-11,
                        'fatol': 1e-11,
                        'maxiter': 50,
                        'adaptive': True
                    }
                )

                if np.isfinite(res_nm2.fun) and res_nm2.fun < self.best_f:
                    self.best_f = res_nm2.fun
                    self.best_x = np.clip(res_nm2.x, -5, 5)

            except StopIteration:
                pass
            except Exception:
                pass

        # Fallback if no calls were made
        if self.best_x is None:
            x0 = np.ones(self.dim)
            try:
                self.best_f = wrapped_func(x0)
                self.best_x = x0.copy()
            except Exception:
                self.best_x = x0
                self.best_f = 1e10

        return self.best_x, self.best_f
