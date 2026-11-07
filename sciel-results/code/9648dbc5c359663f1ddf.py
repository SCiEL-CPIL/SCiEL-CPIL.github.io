"""Deterministic hierarchical grid search exploiting Katsuura fractal structure."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hierarchical grid-based search for Katsuura function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Track best point and function value
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid 3^5 = 243 evaluations
        # This gives a rough picture of the landscape
        grid_sizes = [3, 5, 7, 9, 11]
        calls_made = 0

        # Start with the coarsest grid
        # For each level, we refine around the best region found

        # Level 0: Full domain, 3^5 grid
        lo = np.full(dim, -5.0)
        hi = np.full(dim, 5.0)

        for gsize in grid_sizes:
            # Generate grid points in [lo, hi]^dim
            # We need to be careful about budget
            n_points = gsize ** dim
            if calls_made + n_points > budget:
                # Can't afford this full grid, break
                break

            # Create grid
            # For efficiency, iterate through all combinations
            # Use np.meshgrid for small grids
            axes = [np.linspace(lo[i], hi[i], gsize) for i in range(dim)]
            grids = np.meshgrid(*axes, indexing='ij')

            # Evaluate all grid points
            for idx in np.ndindex(*([gsize] * dim)):
                x = np.array([grids[i][idx] for i in range(dim)])
                evaluate(x)
                calls_made += 1

            if calls_made >= budget:
                break

            # If we have budget left, refine around the best point
            if best_x is None:
                break

            # Determine refinement box: half-width around best
            # Shrink the search region to focus on the promising area
            # Use a fraction of the current box size
            shrink = 0.5
            new_lo = best_x - (hi - lo) * shrink / 2
            new_hi = best_x + (hi - lo) * shrink / 2
            # Clip to original domain
            new_lo = np.clip(new_lo, -5.0, 5.0)
            new_hi = np.clip(new_hi, -5.0, 5.0)

            # Ensure new_lo < new_hi
            for i in range(dim):
                if new_lo[i] >= new_hi[i]:
                    new_lo[i] = best_x[i] - 0.1
                    new_hi[i] = best_x[i] + 0.1

            lo = new_lo
            hi = new_hi

        # Phase 2: Use remaining budget for local refinement with a simple local search
        if best_x is not None and calls_made < budget:
            # Local search: try small perturbations around best_x
            remaining = budget - calls_made
            # Adaptive step size local search
            step = 0.1
            local_x = best_x.copy()
            local_f = best_f

            # Try coordinate-wise and random directions
            rng = np.random.default_rng(42)

            for _ in range(remaining):
                # Mix of coordinate moves and random moves
                if rng.random() < 0.5:
                    # Coordinate move
                    coord = rng.integers(0, dim)
                    direction = rng.choice([-1, 1])
                    move = np.zeros(dim)
                    move[coord] = direction * step
                else:
                    # Random direction
                    move = step * rng.standard_normal(dim)

                y = np.clip(local_x + move, -5.0, 5.0)
                fy = func(y)
                calls_made += 1

                if fy < local_f:
                    local_x = y
                    local_f = fy
                    # Increase step slightly on success
                    step *= 1.3
                else:
                    step *= 0.7

                if fy < best_f:
                    best_f = fy
                    best_x = local_x.copy()

                if step < 1e-8:
                    # Reset step to a small value to continue exploring
                    step = 0.01

        return best_x, best_f
# EVOLVE-BLOCK-END
