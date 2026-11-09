"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start global-to-local hybrid: coarse grid + adaptive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s

        rng = np.random.default_rng(42)

        # Phase 1: Coarse global grid to locate the basin
        # Weierstrass f16 is highly multimodal; a grid helps find the shifted optimum
        # Use a 3^5 = 243 point grid over [-4, 4]^5 (where optimum lies)
        # This is within budget (243 < 500) and gives good coverage
        grid_vals = np.linspace(-4.0, 4.0, 3)  # -4, 0, 4
        grid_points = list(np.meshgrid(*([grid_vals] * dim), indexing='ij'))
        grid_points = np.array([p.flatten() for p in zip(*grid_points)])
        
        # Shuffle grid points to avoid bias in early AOCC
        rng.shuffle(grid_points)
        
        best_x = None
        best_fx = np.inf
        calls = 0

        # Evaluate grid points
        for p in grid_points:
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            fp = func(p)
            calls += 1
            if fp < best_fx:
                best_fx = fp
                best_x = p.copy()

        # Phase 2: Adaptive (1+1)-ES refinement from best grid point
        if best_x is None:
            # Fallback: random start
            best_x = rng.uniform(-5, 5, dim)
            best_fx = func(best_x)
            calls += 1

        x = best_x.copy()
        fx = best_fx
        sigma = 0.5  # Start with smaller step for local refinement

        # Adaptive step size with periodic restarts
        consecutive_failures = 0
        max_failures_before_restart = 30

        while calls < budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fy < best_fx:
                    best_x, best_fx = y.copy(), fy
                sigma *= 1.3  # Expand step on success
                consecutive_failures = 0
            else:
                sigma *= 0.85  # Shrink step on failure
                consecutive_failures += 1
                # Periodic restart if stuck
                if consecutive_failures >= max_failures_before_restart:
                    # Restart from a point near the best found
                    x = best_x + 0.5 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fy < best_fx:
                        best_x, best_fx = x.copy(), fx
                    sigma = 0.3
                    consecutive_failures = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
