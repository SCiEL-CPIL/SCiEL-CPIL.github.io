import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_F = []
        eval_count = 0
        
        # 1. Determine fixed values for non-critical variables
        # Critical: x[0], x[1]
        # Semi-critical: x[2] (hint says fix to 0.75 range)
        # Bulk: x[3]...x[11] (fix to midpoint)
        
        x_fixed = np.zeros(self.dim)
        # x[0] and x[1] will be varied
        # x[2]
        x_fixed[2] = lo[2] + 0.75 * (hi[2] - lo[2])
        # x[3] to x[11]
        for i in range(3, self.dim):
            x_fixed[i] = (lo[i] + hi[i]) / 2.0
            
        # 2. Grid search on x[0] and x[1]
        # Use a slightly denser grid to better capture the front initially
        n_grid = 14
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        grid_points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = x_fixed.copy()
                x[0] = x0
                x[1] = x1
                
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                f = problem(x)
                all_F.append(f)
                grid_points.append(x.copy())
                eval_count += 1
            if eval_count >= self.budget or time.time() > deadline:
                break
        
        if eval_count >= self.budget or time.time() > deadline:
            return np.array(all_F)

        # 3. Local Refinement
        # Find non-dominated points from the grid
        F_grid = np.array(all_F)
        n_points = len(F_grid)
        
        # Simple non-dominated check
        non_dom_mask = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            if not non_dom_mask[i]:
                continue
            for j in range(n_points):
                if i == j or not non_dom_mask[j]:
                    continue
                # If j dominates i, mark i as dominated
                if np.all(F_grid[j] <= F_grid[i]) and np.any(F_grid[j] < F_grid[i]):
                    non_dom_mask[i] = False
                    break
        
        non_dom_indices = np.where(non_dom_mask)[0]
        
        # If we have few non-dominated points, we can refine them more
        # Use remaining budget
        remaining_budget = self.budget - eval_count
        
        # Shuffle non-dominated points to pick a subset if too many
        # However, we want to be systematic. Let's sort by the first objective to ensure coverage
        if len(non_dom_indices) > 0:
            # Sort indices by the first objective value to ensure we refine across the front
            non_dom_indices = non_dom_indices[np.argsort(F_grid[non_dom_indices, 0])]
            
            # We have ~56 evals left (200 - 196).
            # Let's try to refine each non-dominated point with a few steps.
            # Or just do random local searches.
            
            # Strategy: For each non-dominated point, try a few random perturbations.
            # Number of points to refine: min(len(non_dom_indices), max(1, remaining_budget // 5))
            # With 14x14=196 grid points, non_dom will be small (maybe 10-20).
            # remaining_budget is 4. So we can only refine 0-1 points with 5 steps each?
            # No, 4 evals left. So we can do 4 single-step refinements.
            
            # Let's use a more efficient refinement:
            # Pick the top non-dominated points (by hypervolume contribution or just evenly spaced)
            # and do 1-2 steps of local search for each.
            
            # Since budget is tight, let's just do a few random perturbations around the best points.
            # Or better, use the remaining budget to do a finer grid in a promising region?
            # No, local search is better.
            
            # Let's refine the points that are most "central" on the front.
            # Or just take the first few in the sorted list.
            
            n_to_refine = min(len(non_dom_indices), remaining_budget)
            
            for idx in non_dom_indices[:n_to_refine]:
                if time.time() > deadline:
                    break
                if eval_count >= self.budget:
                    break
                    
                x_best = grid_points[idx]
                f_best = F_grid[idx]
                
                # Local search with a small step
                # Try a few directions
                # Since we have very few evals left, let's just do one perturbation per point.
                # Or maybe 2 if budget allows.
                
                # Let's try a small Gaussian perturbation
                step_size = 0.02 * (hi - lo)
                delta = np.random.randn(self.dim) * step_size
                x_new = x_best + delta
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                all_F.append(f_new)
                eval_count += 1
                
                # If we have more budget, we could do more steps, but for now 1 step per point is enough.
                # If the new point is better, we could update x_best and do another step,
                # but we don't have the budget for that typically.

        return np.array(all_F)
