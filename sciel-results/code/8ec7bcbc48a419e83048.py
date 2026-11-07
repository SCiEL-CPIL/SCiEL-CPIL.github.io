"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical grid subdivision followed by local Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0

        def f(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            calls += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # Phase 1: Hierarchical grid subdivision
        # Start with the full box [-5, 5]^dim
        # Subdivide into 2^dim sub-cells, evaluate centers, keep best
        # Repeat until budget is nearly exhausted or time is up

        # We'll use a fixed number of subdivision levels, but stop early if budget/time runs out
        # For dim=5, 2^5=32 evaluations per level. With budget=500, we can do ~15 levels.
        # But we should stop when the cell size is small enough and switch to polish.

        low = np.full(dim, -5.0)
        high = np.full(dim, 5.0)
        center = (low + high) / 2.0
        best_x = center.copy()
        best_f = f(best_x)
        calls_used = 1

        # Number of subdivisions to attempt; we'll stop based on budget and time
        # Each level costs 2^dim evaluations
        num_subdivisions = 2 ** dim  # 32 for dim=5

        # We'll do adaptive: keep subdividing while we have budget and time
        # Stop when cell size is below a threshold or budget is low
        max_levels = 12  # safety cap
        for level in range(max_levels):
            if time.time() > deadline - 5.0:
                break
            if calls_used + num_subdivisions > budget - 20:
                break
            # Subdivide current cell
            # Generate 2^dim sub-cell centers
            # For each dimension, the sub-cell centers are at low + (i + 0.5) * (high - low) / 2
            # where i in {0, 1}
            # So we generate all combinations

            # Build grid of centers
            # Each dimension has 2 points: low + 0.25*span and low + 0.75*span
            spans = high - low
            # For each dim, the two midpoints of the two halves
            half_spans = spans / 2.0
            # Point 0: low + 0.5 * half_span = low + span/4
            # Point 1: low + 1.5 * half_span = low + 3*span/4
            p0 = low + 0.25 * spans
            p1 = low + 0.75 * spans

            # Generate all 2^dim combinations
            # Use itertools.product equivalent
            indices = np.array(np.meshgrid(*[np.array([0, 1])] * dim, indexing='ij')).reshape(dim, -1).T
            # indices has shape (2^dim, dim), each entry is 0 or 1
            # For each combination, center = p0 where index=0, p1 where index=1
            centers = np.where(indices == 0, p0, p1)
            # centers has shape (2^dim, dim)

            # Evaluate all centers
            level_best_x = best_x
            level_best_f = best_f
            for cx in centers:
                if time.time() > deadline - 5.0:
                    break
                if calls >= budget:
                    break
                fc = f(cx)
                if fc < level_best_f:
                    level_best_f = fc
                    level_best_x = cx.copy()

            if level_best_f < best_f:
                best_x = level_best_x
                best_f = level_best_f

            # Update low and high to the best sub-cell
            # Determine which sub-cell contains level_best_x
            # For each dim, if level_best_x[d] < (low[d] + high[d])/2, it's in the lower half
            mid = (low + high) / 2.0
            for d in range(dim):
                if level_best_x[d] < mid[d]:
                    high[d] = mid[d]
                else:
                    low[d] = mid[d]

            # Check if cell is small enough to stop
            cell_size = np.max(high - low)
            if cell_size < 1e-3:
                break

        # Phase 2: Local polish using Nelder-Mead
        if calls < budget - 10 and time.time() < deadline - 5.0:
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead for local polish
                # Set initial simplex around best_x
                x0 = best_x.copy()
                # Bounds
                bounds = [(-5.0, 5.0)] * dim
                # Use a small initial step
                options = {'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False}
                result = minimize(f, x0, method='Nelder-Mead', options=options)
                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun
            except Exception:
                pass

        # Ensure we haven't exceeded budget
        return best_x, best_f
# EVOLVE-BLOCK-END
