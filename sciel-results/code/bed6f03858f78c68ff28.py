import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 280
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if self.calls >= budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if not np.isfinite(f):
                return f
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Use differential evolution for global exploration, then local refinement
        from scipy.optimize import differential_evolution, minimize

        def bounds_func():
            return [(-5, 5)] * dim

        # Phase 1: Differential evolution for global search
        if self.calls < budget - 50 and time.time() < self.deadline - 60:
            try:
                remaining_calls = budget - self.calls - 50
                # DE uses pop_size * (1 + 1) calls per generation
                # Be conservative with maxiter
                maxiter_de = max(5, min(50, remaining_calls // 30))
                popsize = max(10, min(30, remaining_calls // (2 * maxiter_de)))

                res = differential_evolution(
                    wrapped_func,
                    bounds_func(),
                    maxiter=maxiter_de,
                    popsize=popsize,
                    tol=1e-8,
                    polish=False,
                    seed=42,
                    init='sobol'
                )
                if np.isfinite(res.fun) and res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 2: Local refinement with Nelder-Mead from best point
        if self.calls < budget - 20 and time.time() < self.deadline - 30:
            try:
                remaining = budget - self.calls - 20
                maxiter = max(1, min(100, remaining // 3))
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if np.isfinite(res.fun) and res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: Coordinate descent for fine-tuning
        if self.calls < budget - 10 and time.time() < self.deadline - 15:
            try:
                x = self.best_x.copy()
                f = self.best_f
                step = 0.1
                for round_num in range(3):
                    for coord in range(dim):
                        if self.calls >= budget - 5:
                            break
                        if time.time() > self.deadline - 10:
                            break
                        for _ in range(15):
                            if self.calls >= budget - 5:
                                break
                            if time.time() > self.deadline - 10:
                                break
                            x_new = x.copy()
                            x_new[coord] = x[coord] + step
                            f_new = wrapped_func(x_new)
                            if np.isfinite(f_new) and f_new < f:
                                x = x_new
                                f = f_new
                            else:
                                x_new = x.copy()
                                x_new[coord] = x[coord] - step
                                f_new = wrapped_func(x_new)
                                if np.isfinite(f_new) and f_new < f:
                                    x = x_new
                                    f = f_new
                                else:
                                    step *= 0.5
                                    if step < 1e-5:
                                        break
                    if step < 1e-5:
                        break
                if np.isfinite(f) and f < self.best_f:
                    self.best_f = f
                    self.best_x = np.clip(x, -5, 5).copy()
            except Exception:
                pass

        # Phase 4: Final polish
        if self.calls < budget - 5 and time.time() < self.deadline - 5:
            try:
                remaining = budget - self.calls - 5
                if remaining > 10:
                    maxiter = max(1, min(50, remaining // 2))
                    res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                                   options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14,
                                            'adaptive': True})
                    if np.isfinite(res.fun) and res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return self.best_x, self.best_f
