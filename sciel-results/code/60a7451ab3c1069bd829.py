# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Uses Nelder-Mead optimization, robust to rotation and scaling,
    suitable for the Bent Cigar function's anisotropic landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0  # safety margin
        dim = self.dim
        budget = self.budget
        calls = 0

        # Wrapper to count calls and handle any potential issues
        def wrapper(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            # The evaluator clips, but let's ensure we pass a clean array
            # The prompt says func(x) clips, so we can just pass x.
            try:
                return float(func(x))
            except Exception:
                return float('inf')

        # Initial guess: origin
        x0 = np.zeros(dim)
        
        # Use Nelder-Mead
        # maxfev limits the number of function evaluations
        # We reserve some calls for potential restarts or just let it run
        # 450 is safe to stay under 500
        try:
            result = scipy.optimize.minimize(
                wrapper, 
                x0, 
                method='Nelder-Mead',
                options={'maxfev': 450, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            best_x = result.x
            best_f = result.fun
        except Exception:
            # Fallback if minimize fails
            best_x = x0
            best_f = wrapper(x0)
            calls += 1 # wrapper increments, but we need to be careful

        # If we have budget left, maybe try a second run from a different start?
        # Nelder-Mead is local. If we start at 0 and the optimum is at 4, it might work.
        # If the rotation is bad, it might fail.
        # Let's add a simple random restart if we have calls left.
        
        if calls < budget - 50:
            # Try a few random starts
            rng = np.random.default_rng(42)
            current_best_f = best_f
            current_best_x = best_x
            
            for _ in range(5):
                if calls >= budget - 10:
                    break
                x_start = rng.uniform(-5, 5, dim)
                try:
                    res2 = scipy.optimize.minimize(
                        wrapper, 
                        x_start, 
                        method='Nelder-Mead',
                        options={'maxfev': max(10, (budget - calls - 10) // 5), 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    if res2.fun < current_best_f:
                        current_best_f = res2.fun
                        current_best_x = res2.x
                except Exception:
                    pass

            best_x = current_best_x
            best_f = current_best_f

        return best_x, best_f
# EVOLVE-BLOCK-END
