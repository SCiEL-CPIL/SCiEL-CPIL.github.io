"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive Random Search with Nelder-Mead polish for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Adaptive Random Search (ARS)
        # ARS explores broadly and adapts step size based on acceptance rate
        n_ars = 300  # evaluations for ARS phase
        arx = np.zeros(dim)
        arx[0] = 0.0
        arf = counted_func(arx)
        best_f = arf
        best_x = arx.copy()

        # Adaptive step size
        step = 2.0
        # Try a few initial points to get a sense of scale
        for _ in range(10):
            if calls >= budget:
                break
            x_try = rng.uniform(-4, 4, dim)
            f_try = counted_func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
                step = 1.5
            elif f_try < best_f + 10:
                step = max(step * 0.9, 0.3)
            else:
                step = min(step * 1.2, 5.0)

        # Main ARS loop
        stagnation = 0
        for i in range(n_ars - calls):
            if time.time() > deadline - 30.0:
                break
            if calls >= budget:
                break

            # Generate candidate: move from best in random direction
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            x_try = best_x + step * direction
            x_try = np.clip(x_try, -5, 5)
            f_try = counted_func(x_try)

            if f_try < best_f:
                # Accept: move toward this point, maybe increase step
                best_x = x_try.copy()
                best_f = f_try
                stagnation = 0
                step = min(step * 1.1, 4.0)
            else:
                # Reject: shrink step, try from current best
                step = max(step * 0.85, 0.01)
                stagnation += 1

            # Periodic large jumps to escape local minima
            if stagnation > 20:
                x_jump = rng.uniform(-4, 4, dim)
                f_jump = counted_func(x_jump)
                if f_jump < best_f:
                    best_x = x_jump.copy()
                    best_f = f_jump
                    step = 2.0
                    stagnation = 0
                else:
                    step = max(step * 0.9, 0.05)

        # Phase 2: Nelder-Mead polish from best point
        if calls < budget and best_x is not None:
            # Try NM from the best point
            try:
                result = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            # If budget remains, try NM from a few other good points
            if calls < budget - 10:
                # Try from center
                try:
                    result2 = minimize(
                        counted_func,
                        np.zeros(dim),
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if result2.fun < best_f:
                        best_f = result2.fun
                        best_x = result2.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
