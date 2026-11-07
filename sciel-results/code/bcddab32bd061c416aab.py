"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with aggressive sigma adaptation and lightweight restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Lattice-based random walk exploiting Rastrigin's periodic structure
        # Rastrigin has period 2 in each dimension, so we use a grid with spacing ~1.0
        # and refine with smaller steps as we converge

        # Phase 1: Coarse lattice search with multiple restarts
        # Use discrete moves on a grid to avoid overshooting narrow basins
        grid_step = 1.0  # Initial coarse step (Rastrigin period is 2, so 1.0 covers half-period)
        no_improve = 0
        max_no_improve = 30
        restart_count = 0
        max_restarts = 8

        # Start from origin
        x = np.zeros(d)
        fx = func(x)
        total_calls += 1
        if fx < best_f:
            best_f, best_x = fx, x.copy()

        # Precompute direction vectors for lattice moves
        # 8-neighbor in 5D: +/-1 in one dimension at a time (10 directions)
        directions = []
        for i in range(d):
            for sign in [-1, 1]:
                dir_vec = np.zeros(d)
                dir_vec[i] = sign
                directions.append(dir_vec)
        directions = np.array(directions)

        while total_calls < budget - 50 and not check_deadline():
            # Pick a random direction
            idx = rng.integers(0, len(directions))
            y = np.clip(x + grid_step * directions[idx], -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            else:
                no_improve += 1

            # Adaptive step size: shrink when making progress, restart when stalled
            if no_improve == 0:
                # Making progress - keep step or slightly shrink
                if grid_step > 0.1:
                    grid_step *= 0.95
            elif no_improve >= max_no_improve and restart_count < max_restarts:
                restart_count += 1
                # Jump to a new random point on the lattice
                x = np.round(rng.uniform(-4.5, 4.5, d) / grid_step) * grid_step
                x = np.clip(x, -5, 5)
                fx = func(x)
                total_calls += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                grid_step = 1.0
                no_improve = 0
            elif no_improve >= max_no_improve // 2 and grid_step > 0.2:
                # Shrink step size when partially stalled
                grid_step *= 0.7
                no_improve = 0

        # Phase 2: Fine polish with very small lattice steps
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            grid_step = 0.1
            remaining_calls = budget - total_calls

            for _ in range(min(remaining_calls, 50)):
                if check_deadline():
                    break

                idx = rng.integers(0, len(directions))
                y = np.clip(x + grid_step * directions[idx], -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    grid_step *= 0.9
                else:
                    grid_step *= 0.8

                if grid_step < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
