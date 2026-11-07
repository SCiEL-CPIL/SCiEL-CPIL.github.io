"""Optimizer for BBOB f15 (Rastrigin rotated) – discrete lattice walk + multi-start polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer:
      Phase 1: discrete lattice-based random walk with adaptive step size
               to exploit Rastrigin's periodic structure (local minima every ~1 unit).
      Phase 2: continuous (1+1)-ES polish around the best point found.

    Multi-start: several diverse initial points to cover different basins.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # --- Phase 1: discrete lattice walk (multi-start) ---
        # Rastrigin has local minima spaced roughly 1 unit apart in each dimension.
        # Use a discrete step on a grid with spacing ~1.0, then refine.
        n_starts = 8
        # Diverse starting points: origin, corners, random
        starts = [np.zeros(dim)]
        # Add points at various offsets
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Allocate budget: 60% to lattice walk, 40% to polish
        lattice_budget = int(budget * 0.6)
        polish_budget = budget - lattice_budget

        for s_idx, x0 in enumerate(starts):
            x = np.clip(x0, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Discrete lattice walk: 8-neighbor (all ±1 in one coord)
            # Adaptive step size: start at 1.0, shrink if no improvement
            step = 1.0
            no_improve = 0
            max_no_improve = 30

            for _ in range(lattice_budget // n_starts):
                if time.time() > deadline:
                    break

                # Generate a random discrete move: pick one dimension, step ±1
                d = rng.integers(dim)
                direction = rng.choice([-1, 1])
                y = x.copy()
                y[d] += direction * step
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    # Occasionally expand step to explore larger regions
                    if rng.random() < 0.1:
                        step = min(step * 1.5, 3.0)
                else:
                    no_improve += 1
                    # Shrink step when stuck
                    if no_improve >= max_no_improve:
                        step = max(step * 0.5, 0.1)
                        no_improve = 0

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: continuous (1+1)-ES polish ---
        if calls_used < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            polish_calls = 0
            max_polish = polish_budget

            for _ in range(max_polish):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                polish_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    # If sigma gets very small, try a larger jump to escape
                    if sigma < 0.01:
                        sigma = 0.3

        return best_x, best_f
# EVOLVE-BLOCK-END
