import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        eval_count = 0

        def safe_eval(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= total_budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start global exploration targeting Lunacek's two basins
        n_global = min(80, total_budget // 5)
        for i in range(n_global):
            if time.time() > deadline or eval_count >= total_budget:
                break
            if i < 10:
                x = rng.normal(0, 1.0, n)
            elif i < 20:
                x = rng.normal(2.5, 1.0, n)
            elif i < 30:
                x = rng.normal(-2.5, 1.0, n)
            else:
                x = rng.uniform(-5, 5, n)
            safe_eval(x)

        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)

        # Phase 2: Coordinate Descent with line-search-free steps
        remaining = total_budget - eval_count
        if remaining > 0:
            best_x, best_f = self._coordinate_descent(
                safe_eval, best_x, best_f, remaining, n, rng, deadline
            )

        return best_x, best_f

    def _coordinate_descent(self, safe_eval, x0, f0, budget, dim, rng, deadline):
        n = dim
        x = x0.copy()
        fx = f0
        step_size = 2.0
        decay = 0.85
        min_step = 1e-6
        stagnant = 0
        max_stagnant = 15
        calls_made = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            improved_this_cycle = False
            for d in range(n):
                if time.time() > deadline or calls_made >= budget:
                    break

                # Try positive step
                x_pos = x.copy()
                x_pos[d] += step_size
                f_pos = safe_eval(x_pos)
                calls_made += 1
                if f_pos < fx:
                    fx = f_pos
                    x = x_pos.copy()
                    improved_this_cycle = True
                else:
                    # Try negative step
                    x_neg = x.copy()
                    x_neg[d] -= step_size
                    f_neg = safe_eval(x_neg)
                    calls_made += 1
                    if f_neg < fx:
                        fx = f_neg
                        x = x_neg.copy()
                        improved_this_cycle = True

            if not improved_this_cycle:
                stagnant += 1
                if stagnant >= max_stagnant:
                    # Random perturbation to escape local minima
                    perturb = rng.normal(0, step_size * 2, n)
                    x_pert = x + perturb
                    f_pert = safe_eval(x_pert)
                    calls_made += 1
                    if f_pert < fx:
                        fx = f_pert
                        x = x_pert.copy()
                        stagnant = 0
                        step_size *= 1.5
                    else:
                        step_size *= decay
                        if step_size < min_step:
                            step_size = min_step
                else:
                    step_size *= decay
            else:
                stagnant = 0
                step_size *= 0.95

            step_size = max(step_size, min_step)
            step_size = min(step_size, 5.0)

        return x, fx
