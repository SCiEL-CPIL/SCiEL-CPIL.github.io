"""Lattice-based deterministic grid search exploiting Rastrigin periodicity."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Lattice-based deterministic grid search for Rastrigin (f3).
    
    Strategy:
    1. Exploit Rastrigin's periodic structure: f(x) = 10*n + sum((x_i - s_i)^2 - 10*cos(2*pi*(x_i - s_i)))
       Local minima occur at integer offsets from the hidden shift s.
       
    2. Phase 1: Coarse lattice search using integer and half-integer coordinates.
       This identifies the basin containing the global minimum.
       
    3. Phase 2: Fine lattice search around the best point found in Phase 1.
       Uses quarter-integer offsets to narrow down the location.
       
    4. Phase 3: Local (1+1)-ES refinement for final precision.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Coarse lattice search with integer and half-integer coordinates
        # Generate candidate coordinates: integers and half-integers in [-4.5, 4.5]
        # This gives 19 values per dimension: -4.5, -4, -3.5, ..., 3.5, 4, 4.5
        
        # But 19^5 is too many. Use a structured subset.
        # First, evaluate integer grid: 9^5 = 59049, still too many.
        # Use a space-filling subset of the integer grid.
        
        # Phase 1a: Evaluate a sparse integer lattice (every other point in some dims)
        # Use coordinates: -4, -2, 0, 2, 4 (5 values) -> 5^5 = 3125, too many
        # Use: -4, 0, 4 (3 values) -> 3^5 = 243, good for coarse search
        
        coarse_coords = np.array([-4.0, 0.0, 4.0])
        grids = np.meshgrid(*([coarse_coords] * dim), indexing='ij')
        coarse_lattice = np.stack([g.ravel() for g in grids], axis=-1)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate all 243 coarse points
        for x in coarse_lattice:
            if calls_made >= budget - 200:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
        
        # Phase 1b: Fine lattice search around the best coarse point
        # Use half-integer offsets: -0.5, 0, 0.5 in each dimension
        # This gives 3^5 = 243 points around the best coarse point
        
        fine_offsets = np.array([-0.5, 0.0, 0.5])
        fine_grids = np.meshgrid(*([fine_offsets] * dim), indexing='ij')
        fine_lattice = np.stack([g.ravel() for g in fine_grids], axis=-1)
        
        for offset in fine_lattice:
            if calls_made >= budget - 100:
                break
            x = np.clip(best_x + offset, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Even finer lattice search with quarter-integer offsets
        # Use offsets: -0.25, 0, 0.25 in each dimension
        # This gives 3^5 = 243 points
        
        quarter_offsets = np.array([-0.25, 0.0, 0.25])
        quarter_grids = np.meshgrid(*([quarter_offsets] * dim), indexing='ij')
        quarter_lattice = np.stack([g.ravel() for g in quarter_grids], axis=-1)
        
        for offset in quarter_lattice:
            if calls_made >= budget - 50:
                break
            x = np.clip(best_x + offset, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 3: Local (1+1)-ES refinement
        rng = np.random.default_rng(42)
        x = best_x
        fx = best_f
        sigma = 0.1  # Start small since we're already near the minimum
        
        remaining_budget = budget - calls_made
        for _ in range(remaining_budget):
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.95
            if sigma < 1e-8:
                break
        
        return x, fx
# EVOLVE-BLOCK-END
