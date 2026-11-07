"""Harmonic analysis + local search optimizer for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Harmonic analysis approach:
    1. Evaluate function at integer lattice points in [-4, 4]^5 to identify
       the global minimum's integer basin (exploiting Rastrigin's structure).
    2. Perform high-precision local search around the best integer point found.
    3. If budget remains, do secondary searches around other promising integer points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls_made = 0
        max_calls = self.budget
        dim = self.dim

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= max_calls

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Evaluate integer lattice points to find the best basin.
        # Rastrigin f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global minimum is at an integer point (shifted by hidden transform).
        # We evaluate all integer points in [-4, 4]^5.
        # That's 9^5 = 59049 points, but we only have 500 budget.
        # Strategy: Use a sparse grid first, then refine.
        
        # First, evaluate a coarse grid: integers in {-4, -2, 0, 2, 4}^5 = 5^5 = 3125
        # Still too many. Use even sparser: {-4, 0, 4}^5 = 3^5 = 243 points.
        # Then refine around the best with {-1, 0, 1} offsets.
        
        # Phase 1a: Coarse grid evaluation
        coarse_vals = [-4, -2, 0, 2, 4]
        coarse_points = list(np.meshgrid(*([coarse_vals]*dim), indexing='ij'))
        coarse_points = [p.reshape(-1, dim) for p in coarse_points]
        coarse_points = np.stack(coarse_points, axis=-1).reshape(-1, dim)
        
        # Sort by distance from origin (prioritize center)
        dists = np.linalg.norm(coarse_points, axis=1)
        order = np.argsort(dists)
        coarse_points = coarse_points[order]
        
        # Evaluate up to 200 points from the coarse grid
        n_coarse = min(200, len(coarse_points), max_calls // 2)
        for i in range(n_coarse):
            if check_deadline():
                break
            x = coarse_points[i]
            evaluate(x)
        
        if best_x is None:
            x = np.zeros(dim)
            evaluate(x)
            return best_x, best_f

        # Phase 1b: Fine grid around the best coarse point
        # Evaluate neighbors of the best point found in coarse search
        fine_offsets = list(np.ndindex(*([( -1, 0, 1)]*dim)))
        fine_points = []
        for offset in fine_offsets:
            p = best_x + np.array(offset, dtype=float)
            if np.all(p >= -4.5) and np.all(p <= 4.5):
                fine_points.append(p)
        
        # Also add the best point itself (already evaluated)
        fine_points = np.array(fine_points)
        
        # Sort by distance from best_x
        if len(fine_points) > 0:
            fine_dists = np.linalg.norm(fine_points - best_x, axis=1)
            fine_order = np.argsort(fine_dists)
            fine_points = fine_points[fine_order]
            
            n_fine = min(100, len(fine_points), max_calls - calls_made)
            for i in range(n_fine):
                if check_deadline():
                    break
                x = fine_points[i]
                evaluate(x)

        # Phase 2: Local search using (1+1)-ES around the current best
        def local_search(x_start, f_start, budget_left, sigma_init, rng_local):
            nonlocal calls_made, best_x, best_f
            
            x = x_start.copy()
            f = f_start
            sigma = sigma_init
            
            for i in range(budget_left):
                if check_deadline():
                    break
                
                step = sigma * rng_local.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < f:
                    x = y
                    f = fy
                    sigma *= 1.05
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    sigma *= 0.95
                
                sigma = np.clip(sigma, 0.001, 2.0)

            return x, f

        remaining = max_calls - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Phase 2a: High-precision local search
        budget_local = int(remaining * 0.7)
        if budget_local > 0:
            local_search(best_x, best_f, budget_local, sigma_init=0.3, rng_local=rng)
        
        remaining = max_calls - calls_made
        if remaining <= 0 or check_deadline():
            return best_x, best_f

        # Phase 2b: Secondary search if we have budget left
        # Try a perturbation to escape potential local minima
        budget_secondary = remaining
        if budget_secondary > 10:
            x_perturb = best_x + 0.5 * rng.standard_normal(dim)
            x_perturb = np.clip(x_perturb, -5, 5)
            f_perturb = func(x_perturb)
            calls_made += 1
            
            if f_perturb < best_f:
                best_f = f_perturb
                best_x = x_perturb.copy()
            
            local_search(best_x, best_f, budget_secondary - 1, sigma_init=0.2, rng_local=rng)

        return best_x, best_f

# EVOLVE-BLOCK-END
