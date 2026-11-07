"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Rastrigin-aware hybrid: structured init + multi-restart local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)
        
        calls_made = 0
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

        # Phase 1: Rastrigin-aware initial exploration
        # Rastrigin has periodic structure with local minima at multiples of pi.
        # With hidden shift/rotation, the minima are shifted but periodicity is preserved.
        # Sample near the Rastrigin grid with noise to hit basins efficiently.
        
        # Generate candidate points: origin + grid points near multiples of pi with noise
        n_init = 80
        # First: origin
        counted_func(np.zeros(dim))
        
        # Grid points: for each dimension, try values near k*pi for k in {-1, 0, 1}
        # This gives 3^5 = 243 possible grid points, but we sample a subset
        grid_vals = [-np.pi, 0.0, np.pi]
        
        # Use a stratified approach: sample from the grid with random noise
        import itertools
        grid_points = list(itertools.product(grid_vals, repeat=dim))
        
        # Shuffle and take up to n_init-1 points
        rng.shuffle(grid_points)
        
        for i in range(min(n_init - 1, len(grid_points))):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            # Add small noise to avoid exactly hitting the grid
            x = np.array(grid_points[i]) + rng.normal(0, 0.3, dim)
            x = np.clip(x, -5, 5)
            counted_func(x)
        
        # Phase 2: Local Refinement with multiple restarts
        # Use Nelder-Mead for faster convergence than Powell on smooth-ish landscapes
        
        # Track best points for restarts
        # We'll do a few rounds of local search with decreasing radius
        
        radii = [1.5, 0.8, 0.4, 0.2, 0.1]
        for radius in radii:
            if calls_made >= budget or time.time() > deadline - 10:
                break
            
            # Try 2 restarts from perturbations of the current best
            for _ in range(2):
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                
                x_start = best_x + rng.normal(0, radius, dim)
                x_start = np.clip(x_start, -5, 5)
                
                try:
                    res = minimize(
                        counted_func, 
                        x_start, 
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'maxfev': 80, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                except Exception:
                    pass
        
        # Phase 3: Final high-precision polish
        if calls_made < budget and time.time() < deadline - 5:
            try:
                res = minimize(
                    counted_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'maxfev': min(100, budget - calls_made), 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
