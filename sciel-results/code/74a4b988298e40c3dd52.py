import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            if calls_used >= budget or check_deadline():
                return np.inf
            fx = func(x)
            calls_used += 1
            return fx

        # Phase 1: Multi-start (1+1)-ES with restarts
        # Allocate ~55% of budget to global search
        global_budget = int(budget * 0.55)
        max_restarts = 8
        budget_per_restart = max(30, global_budget // max_restarts)

        for restart in range(max_restarts):
            if check_deadline() or calls_used >= global_budget:
                break

            # Initialize: first restart at origin, others random or near best
            if restart == 0:
                x = np.zeros(n)
            elif restart == 1:
                x = rng.uniform(-4, 4, n)
            elif restart == 2:
                # Try points near where Rosenbrock minima often are (before rotation)
                x = rng.uniform(0.5, 1.5, n)
            elif restart == 3:
                x = rng.uniform(-1.5, -0.5, n)
            else:
                # Mix of local jumps around best and global random
                if rng.random() < 0.5:
                    x = best_x + 2.0 * rng.standard_normal(n)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-4, 4, n)

            fx = do_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.5
            min_sigma = 1e-8
            max_sigma = 8.0

            # (1+1)-ES loop for this restart
            steps_in_restart = 0
            max_steps = budget_per_restart
            while steps_in_restart < max_steps:
                if check_deadline() or calls_used >= global_budget:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = do_call(y)
                steps_in_restart += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, max_sigma)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.85, min_sigma)

        # Phase 2: Local polish using Nelder-Mead from multiple starting points
        # Allocate remaining budget to local refinement
        remaining_budget = budget - calls_used
        if remaining_budget > 20 and not check_deadline():
            # Use Nelder-Mead from multiple starting points near best
            polish_starts = [best_x.copy()]
            # Add nearby points for robustness
            for scale in [0.2, 0.5, 1.0, 2.0]:
                px = best_x + scale * rng.standard_normal(n)
                px = np.clip(px, -5, 5)
                polish_starts.append(px)

            for ps in polish_starts:
                if check_deadline() or calls_used >= budget - 5:
                    break

                remaining = budget - calls_used
                if remaining < 15:
                    break

                def objective(x):
                    nonlocal calls_used
                    if check_deadline() or calls_used >= budget:
                        raise ValueError("budget exceeded")
                    val = func(np.clip(x, -5, 5))
                    calls_used += 1
                    return val

                try:
                    # Nelder-Mead is derivative-free, good for Rosenbrock valley
                    res = minimize(
                        objective,
                        ps,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining - 5, 400),
                            'xatol': 1e-9,
                            'fatol': 1e-11,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # If we haven't made any calls (shouldn't happen), make one
        if calls_used == 0:
            x = np.zeros(n)
            fx = do_call(x)
            best_x = x
            best_f = fx

        return best_x, best_f
