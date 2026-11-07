"""Lattice-based deterministic grid search exploiting Rastrigin periodicity."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Lattice-based deterministic grid search for Rastrigin (f3).
    
    Strategy:
    1. Phase 1: Global Basin Identification using a Sobol sequence mapped to a half-integer lattice.
       - Generate 250 points using a Sobol quasi-random sequence in [0, 1)^5.
       - Scale to [-4, 4]^5.
       - Snap each coordinate to the nearest multiple of 0.5.
       - This creates a deterministic, space-filling set of points aligned with the periodic structure
         of the Rastrigin function (period 1.0), ensuring we probe the "wells" effectively.
       
    2. Phase 2: Local Refinement using a (1+1)-ES.
       - Start from the best point found in Phase 1.
       - Use a small initial sigma (0.2) to converge quickly to the local minimum.
       - Adapt sigma based on success/failure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Sobol-based lattice search
        # Reserve 400 calls for global search, 100 for local refinement
        # Increasing global search improves the chance of finding the true global basin
        n_global = 400
        n_local = budget - n_global
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=dim, scramble=False)
        sobol_points = sampler.random(n_global)
        
        # Scale from [0, 1) to [-4, 4]
        # Center is 0, range is 8.
        # x = -4 + 8 * u
        lattice_points = -4.0 + 8.0 * sobol_points
        
        # Snap to nearest 0.5 (half-integer lattice)
        # This aligns with the periodicity of cos(2*pi*x)
        lattice_points = np.round(lattice_points * 2.0) / 2.0
        
        # Ensure points are within [-5, 5] (they should be, as [-4,4] is inside)
        lattice_points = np.clip(lattice_points, -5.0, 5.0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate global lattice points
        for i in range(n_global):
            if calls_made >= n_global:
                break
            x = lattice_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local refinement
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
            
        # Initialize (1+1)-ES
        # Use a deterministic seed for reproducibility
        rng = np.random.default_rng(42)
        x = best_x
        fx = best_f
        # Start with a larger sigma to escape shallow local minima if the global search
        # landed in a nearby but not optimal basin, or to cover the gap to the true min.
        # Since we snapped to 0.5, the true min is likely within 0.5.
        sigma = 0.5
        
        for _ in range(n_local):
            if calls_made >= budget:
                break
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.05  # Slightly slower expansion to be more stable
            else:
                sigma *= 0.95  # Slightly slower contraction
            if sigma < 1e-8:
                break
                
        return x, fx
# EVOLVE-BLOCK-END
