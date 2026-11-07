import numpy as np
import time
from scipy.optimize import dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        rng = np.random.default_rng(42)

        # 1. Initial Global Exploration
        n_init = max(5, int(budget * 0.05)) # 5% of budget for init
        for _ in range(n_init):
            if n_calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            f = timed_call(x)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = timed_call(best_x)

        # 2. Dual Annealing
        # Use the best found so far as the starting point
        # Set maxfev to the remaining budget
        remaining_budget = budget - n_calls
        
        if remaining_budget > 10 and time.time() < deadline:
            try:
                result = dual_annealing(
                    func=lambda x: timed_call(x),
                    x0=best_x,
                    bounds=[(-5.0, 5.0)] * dim,
                    maxfev=remaining_budget,
                    local_optimizer='L-BFGS-B',
                    initial_temp=1.0,
                    restart_temp=1.0,
                    niter_no_improve=10,
                    seed=42
                )
                # The result object contains the best point and value
                # But we already track best_x and best_f in timed_call
            except Exception:
                # If dual_annealing fails, we just use what we have
                pass

        # 3. Final Polish with Nelder-Mead if budget remains?
        # Dual annealing should use up most of it.
        # If there are leftover calls, do a few random steps or Nelder-Mead
        if n_calls < budget and time.time() < deadline:
            # Simple local refinement
            step_size = 0.1
            for _ in range(budget - n_calls):
                if time.time() > deadline:
                    break
                x = np.clip(best_x + step_size * rng.standard_normal(dim), -5.0, 5.0)
                f = timed_call(x)
                if f < best_f:
                    # Shrink step size if we improved? Or just keep going
                    step_size *= 0.9
                else:
                    step_size *= 1.1
                    step_size = min(step_size, 1.0)

        return best_x, best_f
