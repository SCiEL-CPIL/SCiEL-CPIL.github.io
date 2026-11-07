"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Grid-based adaptive search with recursive subdivision for Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Coarse grid search to identify the best region
        # Start with a 2^dim grid over [-5, 5]^dim
        # For dim=5, that's 32 evaluations
        n_coarse = 2 ** dim  # 32 for dim=5
        if n_coarse <= budget:
            # Generate coarse grid points
            grid_bounds = [(-5.0, 5.0)] * dim
            # Create 2^dim points by splitting each dimension into 2
            step_size = (5.0 - (-5.0)) / 2.0  # 5.0
            for i in range(n_coarse):
                if calls_made >= budget or time.time() > deadline - 30:
                    break
                # Convert index to grid coordinates
                coords = []
                idx = i
                for d in range(dim):
                    # Use binary representation to determine which half
                    bit = (idx >> (dim - 1 - d)) & 1
                    coord = -5.0 + bit * step_size
                    coords.append(coord)
                x = np.array(coords)
                counted_func(x)

        # Phase 2: Recursive adaptive subdivision
        # Subdivide the best cell found in Phase 1
        # Track the best cell center and its bounds
        best_cell_center = best_x.copy()
        cell_size = 10.0 / 2.0  # Initial cell size after first subdivision
        
        # We'll do several rounds of subdivision
        max_subdivision_rounds = 6
        for round_idx in range(max_subdivision_rounds):
            if calls_made >= budget or time.time() > deadline - 30:
                break
            
            # Determine how many sub-cells we can afford
            # Each subdivision round: evaluate 2^dim points in the current best cell
            n_sub = 2 ** dim  # 32 for dim=5
            
            if calls_made + n_sub > budget:
                # Can't afford full subdivision, do what we can
                n_sub = budget - calls_made
                if n_sub <= 0:
                    break
            
            # Subdivide the best cell
            # Current cell is centered at best_cell_center with size cell_size
            half_size = cell_size / 2.0
            
            # Generate sub-cell centers
            for i in range(n_sub):
                if calls_made >= budget or time.time() > deadline - 30:
                    break
                # Convert index to sub-cell coordinates
                coords = []
                idx = i
                for d in range(dim):
                    bit = (idx >> (dim - 1 - d)) & 1
                    # Offset from center
                    offset = -half_size / 2.0 + bit * half_size
                    coord = best_cell_center[d] + offset
                    coord = clip(coord)
                    coords.append(coord)
                x = np.array(coords)
                fx = counted_func(x)
            
            # Update cell size for next round
            cell_size = half_size
        
        # Phase 3: Local refinement using Powell on the best point
        if calls_made < budget and time.time() < deadline - 10:
            try:
                minimize(
                    counted_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 50, 'maxfev': 50}
                )
            except Exception:
                pass
        
        # Phase 4: Final Nelder-Mead polish
        if calls_made < budget and time.time() < deadline - 5:
            try:
                minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
