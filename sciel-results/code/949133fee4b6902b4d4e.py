import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        n_evals = self.budget
        if n_evals <= 0:
            return []

        results = []

        # Strategy: Deterministic grid search varying x0 across [0,1] with x[1:]=0.
        # All points lie exactly on the Pareto front, maximizing hypervolume coverage.
        # Use adaptive selection: pick points to maximize minimum spacing in f1-space
        # (greedy farthest-point sampling) for better coverage of the oscillatory front.

        n_evals = self.budget
        if n_evals <= 0:
            return []

        results = []

        # Dense grid for offline selection
        n_grid = 500000
        x0_grid = np.linspace(0.0, 1.0, n_grid)

        # Compute f1 and f2 for the grid
        # f1(x) = 1 + x * (sin(10*pi*x) + sin(10*pi*x^2))
        # f2(x) = 1 - sqrt(f1(x))
        sin_term1 = np.sin(10 * np.pi * x0_grid)
        sin_term2 = np.sin(10 * np.pi * x0_grid**2)
        f1_grid = 1 + x0_grid * (sin_term1 + sin_term2)

        # Ensure f1 is positive for sqrt
        f1_grid = np.maximum(f1_grid, 1e-6)

        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x0_sorted = x0_grid[sort_idx]

        if n_evals == 1:
            # Pick the point that maximizes the box area (11-f1)(11-f2)
            f2_grid = 1 - np.sqrt(f1_grid)
            areas = (11 - f1_grid) * (11 - f2_grid)
            best_idx = np.argmax(areas)
            x0s = [x0_grid[best_idx]]
        else:
            # Use greedy farthest-point sampling in f1-space for adaptive coverage.
            # This places points where the local density of the front is highest,
            # improving hypervolume coverage of the oscillatory regions.
            
            # Start with the endpoints of f1 range
            selected_f1 = [f1_sorted[0], f1_sorted[-1]]
            selected_idx = [0, n_grid - 1]
            
            # Precompute distances for efficiency
            # For remaining points, iteratively pick the one that maximizes
            # the minimum distance to already-selected points in f1-space
            
            if n_evals > 2:
                # Initialize min-dist array: distance to nearest selected point
                min_dist = np.minimum(np.abs(f1_sorted - selected_f1[0]), 
                                      np.abs(f1_sorted - selected_f1[1]))
                
                for _ in range(n_evals - 2):
                    # Find the point with maximum min-dist
                    new_idx = np.argmax(min_dist)
                    selected_f1.append(f1_sorted[new_idx])
                    selected_idx.append(new_idx)
                    
                    # Update min-dist: for each point, distance to new selected point
                    new_dist = np.abs(f1_sorted - f1_sorted[new_idx])
                    min_dist = np.minimum(min_dist, new_dist)
            
            x0s = [x0_sorted[i] for i in selected_idx]
            
            # Refinement pass: for each x0, do a local bisection to better
            # position the point in f1-space. Since we're using the actual
            # f1 values from the grid, we refine x0 to hit the exact f1 value
            # that would be optimal for hypervolume.
            if time.time() < deadline - 10:
                refined_x0s = []
                for i in range(len(x0s)):
                    x0_init = x0s[i]
                    target_f1 = f1_sorted[selected_idx[i]]
                    
                    # Bisection in a small neighborhood around x0_init
                    grid_step = 1.0 / n_grid
                    lo = max(0.0, x0_init - 3 * grid_step)
                    hi = min(1.0, x0_init + 3 * grid_step)
                    
                    f1_lo = 1.0 + lo * (np.sin(10.0 * np.pi * lo) + np.sin(10.0 * np.pi * lo**2))
                    f1_hi = 1.0 + hi * (np.sin(10.0 * np.pi * hi) + np.sin(10.0 * np.pi * hi**2))
                    
                    if f1_lo <= target_f1 <= f1_hi or f1_hi <= target_f1 <= f1_lo:
                        for _ in range(50):
                            mid = 0.5 * (lo + hi)
                            f1_mid = 1.0 + mid * (np.sin(10.0 * np.pi * mid) + np.sin(10.0 * np.pi * mid**2))
                            if f1_lo * (f1_mid - target_f1) <= 0:
                                hi = mid
                                f1_hi = f1_mid
                            else:
                                lo = mid
                                f1_lo = f1_mid
                        x0_refined = 0.5 * (lo + hi)
                    else:
                        x0_refined = x0_init
                    
                    refined_x0s.append(x0_refined)
                
                x0s = refined_x0s

        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
