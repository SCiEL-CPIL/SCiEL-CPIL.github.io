import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 300s limit, keep 20s margin
        rng = np.random.default_rng(42)
        eval_count = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def wrapped(x):
            nonlocal eval_count, best_f, best_x
            if time.time() >= deadline:
                raise TimeoutError("Time limit exceeded")
            if eval_count >= budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5, 5)
            f_val = float(func(x))
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Multi-start Nelder-Mead with limited budget per start
        # Start points: origin, several random points
        starts = [np.zeros(dim)]
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, dim))

        # Allocate budget: first start gets more, others get less
        # We'll do 4 starts with budgets: 120, 100, 100, 80 (total 400)
        # Keep 100 in reserve for polishing
        start_budgets = [120, 100, 100, 80]

        for i, x0 in enumerate(starts):
            if eval_count >= budget or time.time() >= deadline:
                break
            remaining_budget = budget - eval_count
            maxfev = min(start_budgets[i], remaining_budget)
            if maxfev < 10:
                break
            try:
                minimize(
                    wrapped,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': maxfev,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # Phase 2: Polish from the best point found
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            polish_fev = min(remaining, 100)
            if polish_fev >= 5:
                try:
                    minimize(
                        wrapped,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'maxfev': polish_fev,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, ValueError):
                    pass
                except Exception:
                    pass

        # Phase 3: If still have budget, try a few more random starts for robustness
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            if remaining >= 20:
                for _ in range(2):
                    if eval_count >= budget or time.time() >= deadline:
                        break
                    x0 = rng.uniform(-5, 5, dim)
                    maxfev = min(remaining, 50)
                    if maxfev < 10:
                        break
                    try:
                        minimize(
                            wrapped,
                            x0,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 40,
                                'maxfev': maxfev,
                                'xatol': 1e-10,
                                'fatol': 1e-10,
                                'adaptive': True
                            }
                        )
                    except (TimeoutError, ValueError):
                        break
                    except Exception:
                        continue

        # Fallback: ensure at least one evaluation was made
        if eval_count == 0:
            for _ in range(min(10, budget)):
                x = rng.uniform(-5, 5, dim)
                try:
                    f_val = float(func(x))
                    eval_count += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                except Exception:
                    break

        return best_x, best_f
