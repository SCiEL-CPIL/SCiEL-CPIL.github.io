"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic grid-based coarse-to-fine search with systematic Halton sampling."""

    @staticmethod
    def _halton_sequence_fast(n, dim):
        """Faster Halton sequence generation using vectorized operations."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        base = primes[:dim]
        points = np.zeros((n, dim))
        for d in range(dim):
            b = base[d]
            idx = np.arange(1, n + 1)
            f = 1.0 / b
            frac = np.zeros(n)
            i_val = idx.copy()
            while np.any(i_val > 0):
                frac += f * (i_val % b)
                i_val //= b
                f /= b
            points[:, d] = frac
        return points

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.low = -5.0
        self.high = 5.0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        low, high = self.low, self.high

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Coarse-to-fine grid search with Halton sequence
        # Start with a coarse grid, refine by halving step size when improvement found
        
        # Phase 1: Very coarse exploration - sample the full box with Halton
        n_initial = min(50, budget // 6)
        halton = self._halton_sequence_fast(n_initial, dim)
        x_points = low + (high - low) * halton
        
        for i in range(n_initial):
            if calls_made >= budget:
                break
            x = x_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Progressive refinement - narrow around best with halving radius
        # Use multiple refinement passes with decreasing radius
        refinement_budget = budget - calls_made
        if refinement_budget > 0 and best_x is not None:
            # Define refinement stages with decreasing radii
            radii = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
            
            for radius in radii:
                if calls_made >= budget:
                    break
                
                # Calculate how many points to sample at this radius
                # Use more points for larger radii, fewer for smaller
                points_at_radius = max(5, min(20, int(refinement_budget * 0.3)))
                points_at_radius = min(points_at_radius, budget - calls_made)
                
                if points_at_radius <= 0:
                    break
                
                # Generate Halton points in a hypercube centered at best_x
                halton_local = self._halton_sequence_fast(points_at_radius, dim)
                # Map from [0,1)^dim to [-radius, radius]^dim centered at best_x
                x_local = best_x + radius * (2 * halton_local - 1)
                x_local = np.clip(x_local, low, high)
                
                found_improvement = False
                for i in range(points_at_radius):
                    if calls_made >= budget:
                        break
                    x = x_local[i]
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        found_improvement = True
                
                # If no improvement found at this radius, continue to smaller radius
                # If improvement found, we could consider staying or moving to next smaller radius
                # We always move to next smaller radius for systematic refinement
        
        # Phase 3: Final local optimization using coordinate descent with adaptive step
        if calls_made < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.01  # Start with small step for final polishing
            
            # Coordinate descent with adaptive step size
            while calls_made < budget and step > 1e-6:
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        if calls_made >= budget:
                            break
                        y = x.copy()
                        y[d] += direction * step
                        y = np.clip(y, low, high)
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                    if calls_made >= budget:
                        break
                if not improved:
                    step *= 0.5
                else:
                    step *= 0.7  # Shrink step more aggressively when improving
        
        return best_x, best_f
# EVOLVE-BLOCK-END
