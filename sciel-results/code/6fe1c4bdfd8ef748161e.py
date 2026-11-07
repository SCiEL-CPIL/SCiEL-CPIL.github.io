"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-based grid search followed by CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        # Phase 1: Lattice-based deterministic grid search
        # Rastrigin has periodic structure with period 2 in each dimension.
        # We evaluate points at integer and half-integer coordinates within [-5, 5]^5.
        # This gives a grid of 11 points per dimension (from -5 to 5 step 1), 
        # but we also add half-integers for finer coverage: 21 points per dim.
        # However, 21^5 = 4,084,101 which is too many. 
        # Let's use a coarser grid first: integers only (11^5 = 161,051) - still too many.
        # 
        # Better approach: Use a structured subset. Since the shift is hidden but the
        # function is periodic with period 2, evaluating at all integer points in [-4, 4]^5
        # (which is 9^5 = 59049) might work but is still a lot.
        #
        # Let's use a smarter approach: evaluate at a smaller set of strategic points.
        # We'll use a Sobol-like sequence or a stratified grid with fewer points.
        #
        # Actually, let's use a two-level approach:
        # Level 1: Coarse grid with step 2 (period of Rastrigin) - 5^5 = 3125 points
        # Level 2: Refine around the best few points with step 1
        # Level 3: CMA-ES local refinement

        # Level 1: Coarse grid - points at -5, -3, -1, 1, 3, 5 in each dim
        # That's 6^5 = 7776 points. Still a lot for 500 budget.
        # Let's use a random sampling of the lattice instead.

        # Better: Use a stratified random approach on the lattice
        rng = np.random.default_rng(42)

        # Generate candidate points on the lattice
        # Use points at integer and half-integer coordinates
        # For each dimension, candidates are: -5, -4.5, -4, ..., 4.5, 5
        # That's 21 points per dim. We can't evaluate all 21^5.
        #
        # Strategy: Use a space-filling sequence to sample from the lattice
        # We'll generate points at integer coordinates first (11^5 = 161,051)
        # and sample from them, then refine.

        # Actually, with only 500 calls, let's be strategic:
        # Phase 1a: Evaluate at 0, 100, 200 strategic points to find the basin
        # Phase 1b: Refine with CMA-ES

        # Use a deterministic set of points that covers the search space well
        # Combine:
        # 1. Origin and box corners
        # 2. Integer lattice points sampled via Sobol
        # 3. Then CMA-ES

        # Generate Sobol-like points using a simple LHS (Latin Hypercube) approach
        # for the lattice

        # Phase 1: Global exploration with lattice points
        # We'll use 200 calls for global search
        n_global = 200
        n_local = budget - n_global

        # Generate lattice points: integers from -5 to 5
        lattice_vals = np.arange(-5, 6, 1.0)  # 11 values
        # Create all combinations would be 11^5 = 161,051
        # Sample n_global points from these

        # Use a stratified sampling approach
        # For each dimension, divide into 11 strata (one per integer)
        # Sample proportionally

        # Simpler: just use random sampling from the lattice
        # But make it deterministic and structured

        # Use a grid with step 2 first (coarser), then refine
        coarse_vals = np.arange(-5, 6, 2.0)  # -5, -3, -1, 1, 3, 5 -> 6 values
        # 6^5 = 7776 points, sample 100 from these

        # Generate all coarse grid points
        coarse_grid = np.array(np.meshgrid(*[coarse_vals] * dim, indexing='ij')).T.reshape(-1, dim)
        # Shuffle deterministically
        perm = rng.permutation(len(coarse_grid))
        coarse_grid = coarse_grid[perm]

        best_x = None
        best_f = np.inf

        # Evaluate first 100 coarse grid points
        n_coarse = min(100, len(coarse_grid), n_global)
        for i in range(n_coarse):
            if not time_ok():
                break
            x = coarse_grid[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1b: Finer grid around the best point
        # Use step 0.5 around the best point found
        if best_x is not None and time_ok():
            fine_vals = np.arange(best_x[0] - 2, best_x[0] + 2.1, 0.5)
            # Clip to box
            fine_vals = np.clip(fine_vals, -5, 5)
            # Generate fine grid around best point
            # Only refine in a local region
            local_center = best_x
            local_radius = 2.0
            # Generate points on a 0.5-step grid within the local region
            local_points = []
            for d in range(dim):
                local_points.append(np.arange(
                    max(-5, local_center[d] - local_radius),
                    min(5, local_center[d] + local_radius) + 0.1,
                    0.5
                ))
            local_grid = np.array(np.meshgrid(*local_points, indexing='ij')).T.reshape(-1, dim)
            # Shuffle
            perm2 = rng.permutation(len(local_grid))
            local_grid = local_grid[perm2]

            n_fine = min(100, len(local_grid), n_global - n_coarse)
            for i in range(n_fine):
                if not time_ok():
                    break
                x = local_grid[i]
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 2: CMA-ES local refinement
        if best_x is not None and time_ok() and n_local > 0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            cma_calls = 0
            max_cma_calls = min(n_local, 300)

            # Simple (1+1)-ES with adaptive step size
            for _ in range(max_cma_calls):
                if not time_ok() or cma_calls >= n_local:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                cma_calls += 1
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
                    if sigma < 1e-6:
                        break

        # Final evaluation if we haven't used all budget
        if best_x is None:
            # Fallback: evaluate at origin
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
