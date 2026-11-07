import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 285.0

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return float('inf')
            if calls >= self.budget:
                return float('inf')
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
            except Exception:
                val = float('inf')
            if val < best_f:
                best_f = val
                best_x = np.asarray(x, dtype=float).copy()
            return val

        # Initial point: Center of the box is a decent guess for BBOB f8
        # The minimum is shifted, but often near the center or slightly off.
        # Let's try a few random starts.
        
        # Strategy: 
        # 1. Evaluate a few random points to find a good basin.
        # 2. Run Powell from the best point.
        # 3. If budget remains, run Nelder-Mead or another Powell restart.

        # Phase 1: Exploration
        # Rosenbrock is ill-conditioned. Random points might be far away.
        # Let's sample 5 points.
        n_explore = 5
        for i in range(n_explore):
            if calls >= self.budget or time.time() > deadline:
                break
            # Sample in [-5, 5]
            x = rng.uniform(-5, 5, self.dim)
            wrapped_func(x)

        # Phase 2: Exploitation with Powell
        # Powell is derivative-free and handles the valley well.
        if best_x is not None and calls < self.budget and time.time() < deadline:
            # Powell can use many calls. Let's give it a significant chunk.
            # We don't know exactly how many calls Powell will use, so we monitor via wrapped_func.
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 1000, # High limit, wrapped_func will stop it via budget
                        'xtol': 1e-8,
                        'ftol': 1e-10
                    }
                )
            except Exception:
                pass

        # Phase 3: Fine-tuning
        # If budget remains, use Nelder-Mead for local refinement.
        # Nelder-Mead is robust for final steps.
        if best_x is not None and calls < self.budget - 10 and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': self.budget - calls,
                        'xatol': 1e-10,
                        'fatol': 1e-10
                    }
                )
            except Exception:
                pass

        # Phase 4: If we still have budget, try a different starting point for Powell?
        # Rosenbrock has only one minimum, so if Powell converged, it's likely the global min.
        # If it didn't converge, it might be stuck. A restart from a random point might help if the first Powell run was poor.
        if best_x is not None and calls < self.budget - 50 and time.time() < deadline:
             # Only if we haven't reached a very low value (heuristic)
             # Since f_opt is unknown, we can't check absolute value.
             # But if we have used < 80% of budget, we can try one more Powell run from a random point.
             # This is risky if the first run was already good.
             # Let's stick to the single Powell run + Nelder-Mead polish for stability.
             pass

        return best_x, best_f
