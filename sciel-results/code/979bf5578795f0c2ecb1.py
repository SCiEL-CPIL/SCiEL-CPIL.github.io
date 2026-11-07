"""Grid-based decomposition multi-objective optimizer for RE37."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Grid-based decomposition:
    1. Coarse grid evaluation to map the objective landscape.
    2. Identify non-dominated points on the grid.
    3. Local refinement via Nelder-Mead on promising regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Budget allocation:
        # - Use ~60% of budget for coarse grid
        # - Use ~40% for local refinement
        # Grid size: we want a manageable number of grid points.
        # For 4D, if we use g points per dimension, total = g^4.
        # Let's aim for grid_points <= budget * 0.6
        # g^4 <= 0.6 * 200 = 120 -> g <= 3.3 -> g=3 gives 81 points.
        # g=4 gives 256 points, which is too many.
        # Let's try g=3 (81 points) for coarse grid.
        # Remaining budget: 200 - 81 = 119 for refinement.
        
        # Actually, let's be more dynamic.
        # If budget is large, we can afford a finer grid.
        # For dim=4, g=3 is 81, g=4 is 256.
        # Let's stick with g=3 for the coarse grid to be safe.
        
        g = 3
        grid_points_per_dim = g
        total_grid_points = grid_points_per_dim ** dim
        
        # If total_grid_points > budget * 0.5, reduce g
        while total_grid_points > budget * 0.5 and grid_points_per_dim > 2:
            grid_points_per_dim -= 1
            total_grid_points = grid_points_per_dim ** dim
        
        # Generate grid points
        axes = [np.linspace(lo[i], hi[i], grid_points_per_dim) for i in range(dim)]
        grid = np.array(np.meshgrid(*axes)).reshape(dim, -1).T
        
        # Evaluate grid points
        F_grid = np.array([problem(x) for x in grid])
        used = total_grid_points
        
        # Find non-dominated points in the grid
        def is_dominated(F):
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                fi = F[i]
                # Check if any other point dominates fi
                # F[j] <= fi and F[j] < fi
                less_equal = np.all(F <= fi, axis=1)
                less_than = np.any(F < fi, axis=1)
                dominates = less_equal & less_than
                dominates[i] = False
                if np.any(dominates):
                    dominated[i] = True
            return dominated
        
        dom_grid = is_dominated(F_grid)
        non_dom_indices = np.where(~dom_grid)[0]
        
        if len(non_dom_indices) == 0:
            non_dom_indices = np.arange(len(grid))
        
        # Calculate a simple hypervolume contribution score for each non-dominated point
        # to prioritize refinement
        # Reference point for local scoring: [1.1, 1.3, 1.1]
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Sort non-dominated points by their "quality"
        # We want points that are close to the ideal point but not dominated
        # Simple heuristic: sum of objectives
        scores = np.sum(F_grid[non_dom_indices], axis=1)
        sorted_non_dom = non_dom_indices[np.argsort(scores)]
        
        remaining_budget = budget - used
        # Estimate cost per refinement: ~15-20 evaluations
        # We have ~119 budget left. 119 / 15 ~= 8 refinements.
        n_refine = min(8, len(sorted_non_dom), max(1, remaining_budget // 15))
        
        if n_refine > 0:
            selected_indices = sorted_non_dom[:n_refine]
            
            # Define weights for scalarization to guide the search
            # We will use a weighted sum with weights that emphasize different objectives
            # to ensure we cover the front
            weight_sets = [
                np.array([1.0, 1.0, 1.0]),
                np.array([2.0, 1.0, 1.0]),
                np.array([1.0, 2.0, 1.0]),
                np.array([1.0, 1.0, 2.0]),
                np.array([1.5, 1.5, 1.0]),
                np.array([1.5, 1.0, 1.5]),
                np.array([1.0, 1.5, 1.5]),
                np.array([1.0, 1.0, 1.0])
            ]
            
            for i, idx in enumerate(selected_indices):
                if used >= budget:
                    break
                x0 = grid[idx]
                
                # Select a weight set
                w = weight_sets[i % len(weight_sets)]
                
                # Define objective for Nelder-Mead: weighted sum
                def obj(x, w=w):
                    nonlocal used
                    if used >= budget:
                        return np.inf
                    fx = problem(x)
                    used += 1
                    return np.dot(w, fx)
                
                # Nelder-Mead options
                # maxfev should be small to allow multiple refinements
                maxfev = min(20, budget - used)
                if maxfev <= 0:
                    break
                options = {'maxiter': 50, 'maxfev': maxfev}
                
                try:
                    res = minimize(obj, x0, method='Nelder-Mead', bounds=list(zip(lo, hi)), options=options)
                    # Evaluate the result one last time to ensure it's recorded
                    if used < budget:
                        xf = res.x
                        # Ensure xf is within bounds (minimize should handle it, but clip just in case)
                        xf = np.clip(xf, lo, hi)
                        ff = problem(xf)
                        used += 1
                except Exception:
                    pass
        
        return grid
# EVOLVE-BLOCK-END
