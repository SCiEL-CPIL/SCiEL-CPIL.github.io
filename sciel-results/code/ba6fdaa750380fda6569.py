"""Coordinate-wise ladder search for BBOB-style Rastrigin optimization."""
import numpy as np
import time

class Optimizer:
    """
    Deterministic coordinate-wise ladder search exploiting Rastrigin separability.
    
    Strategy:
    1. Coarse grid initialization to find a promising basin.
    2. Sequential coordinate-wise minimization using golden-section search.
    3. Cycle through all dimensions repeatedly, refining the best point.
    4. Exploit the known structure of Rastrigin (sum of 10 + 10*(x^2 - cos(2*pi*x)))
       to guide step sizes and detect local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).copy()
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e6
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid initialization
        # Use a small grid to find a good starting basin
        # For 5D, use 2 points per dim = 32 evaluations
        n_init = min(32, budget)
        if n_init >= 32:
            grid_vals = np.linspace(-4.0, 4.0, 2)  # 2 points per dim
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            for m in range(2):
                                if calls_made >= budget:
                                    break
                                x = np.array([grid_vals[i], grid_vals[j], grid_vals[k], grid_vals[l], grid_vals[m]])
                                counted_func(x)
        else:
            # Fallback: random initialization
            rng = np.random.default_rng(42)
            for _ in range(n_init):
                x = rng.uniform(-4.5, 4.5, dim)
                counted_func(x)

        # Phase 2: Coordinate-wise ladder search
        # Golden-section ratio
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618

        # Step sizes for each coordinate refinement
        # Start with moderate step and shrink
        base_steps = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
        step_idx = 0

        # Number of coordinate sweeps
        max_sweeps = 50
        sweep = 0

        while calls_made < budget and time.time() < deadline and sweep < max_sweeps:
            sweep += 1
            step = base_steps[min(step_idx, len(base_steps) - 1)]

            # For each coordinate, do a local search around current best
            for coord in range(dim):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Golden-section search along this coordinate
                # We'll evaluate a few points to find a better value
                # Try points at best_x +- step, best_x +- step/2, etc.
                
                # First, check if moving in + direction helps
                x_plus = best_x.copy()
                x_plus[coord] = best_x[coord] + step
                x_plus = np.clip(x_plus, -5.0, 5.0)
                f_plus = counted_func(x_plus)

                if calls_made >= budget or time.time() > deadline:
                    break

                x_minus = best_x.copy()
                x_minus[coord] = best_x[coord] - step
                x_minus = np.clip(x_minus, -5.0, 5.0)
                f_minus = counted_func(x_minus)

                if calls_made >= budget or time.time() > deadline:
                    break

                # Determine which direction is better
                if f_plus < best_f and f_plus <= f_minus:
                    # Move in + direction
                    # Do a few more evaluations to refine
                    for sub in range(3):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        new_step = step * (0.5 ** (sub + 1))
                        x_new = best_x.copy()
                        x_new[coord] = best_x[coord] + new_step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        counted_func(x_new)
                elif f_minus < best_f:
                    # Move in - direction
                    for sub in range(3):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        new_step = step * (0.5 ** (sub + 1))
                        x_new = best_x.copy()
                        x_new[coord] = best_x[coord] - new_step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        counted_func(x_new)
                else:
                    # Neither direction helped, try smaller step
                    # Just evaluate a point at half step in the better direction
                    if f_plus <= f_minus:
                        x_half = best_x.copy()
                        x_half[coord] = best_x[coord] + step * 0.5
                        x_half = np.clip(x_half, -5.0, 5.0)
                        counted_func(x_half)
                    else:
                        x_half = best_x.copy()
                        x_half[coord] = best_x[coord] - step * 0.5
                        x_half = np.clip(x_half, -5.0, 5.0)
                        counted_func(x_half)

            # After a full sweep, shrink the step if we're making progress
            # Check if best_f improved significantly
            if step_idx < len(base_steps) - 1:
                # Shrink step for next sweep
                step_idx += 1

            # Occasionally, do a random perturbation to escape local minima
            if sweep % 10 == 0 and calls_made < budget - 20:
                rng = np.random.default_rng(sweep * 1000)
                x_pert = best_x + rng.normal(0, 0.5, dim)
                x_pert = np.clip(x_pert, -5.0, 5.0)
                counted_func(x_pert)

        return best_x, best_f
