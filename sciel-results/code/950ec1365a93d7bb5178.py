"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Two-phase search: Lattice global scan followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        calls_made = 0
        
        # Wrap func to count calls and enforce budget/deadline
        def safe_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls_made += 1
            return func(x)

        best_x = None
        best_f = np.inf
        
        # Phase 1: Coarse Global Lattice Sampling
        # We want to cover [-5, 5]^5. 
        # A lattice of 4 points per dimension gives 4^5 = 1024 points.
        # This is too many for budget 500.
        # A lattice of 3 points per dimension gives 3^5 = 243 points.
        # This fits well within 500 and provides decent coverage.
        # Let's use 3 points per dimension: -5, 0, 5.
        # Actually, to be more robust against shifts, maybe slightly inside: -4, 0, 4?
        # Or just use random sampling if lattice is too rigid?
        # The prompt suggests "Lattice-based sampling".
        # Let's stick to a structured grid.
        # 3 points per dim: [-4, 0, 4] covers the likely minimizer range [-4, 4] mentioned in contract.
        
        grid_points_per_dim = 3
        coords = np.linspace(-4, 4, grid_points_per_dim)
        
        # Generate all combinations
        grid = np.array(np.meshgrid(*([coords] * self.dim))).T.reshape(-1, self.dim)
        
        # Shuffle to avoid bias if we run out of budget (though 243 < 500)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid))
        grid = grid[perm]
        
        # Evaluate grid
        # We have budget 500. Grid is 243.
        # We can evaluate all of them.
        
        for x in grid:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Identify top 3 distinct basins
        # We have evaluated 243 points. We want to pick 3 starting points for Nelder-Mead.
        # Simple strategy: Pick the 3 best points from the grid.
        # To ensure diversity, we can check distance.
        # Let's collect all evaluated points and their values.
        # Since we shuffled, we don't have the original indices easily mapped to values 
        # unless we store them. Let's re-do evaluation storing results.
        
        # Reset for cleaner tracking
        calls_made = 0
        def safe_func_v2(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls_made += 1
            return func(x)
            
        best_x = None
        best_f = np.inf
        eval_results = []
        
        # Re-evaluate grid with tracking
        for x in grid:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            fx = safe_func_v2(x)
            eval_results.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Sort by function value
        eval_results.sort(key=lambda t: t[0])
        
        # Select top 3 distinct points
        # "Distinct" means not too close to each other.
        # In 5D with range 8, a distance of > 1.0 is a good heuristic for distinctness.
        selected_starts = []
        min_dist = 1.5
        
        for fx, x in eval_results:
            if len(selected_starts) >= 3:
                break
            # Check distance against already selected starts
            is_distinct = True
            for _, x_sel in selected_starts:
                if np.linalg.norm(x - x_sel) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                selected_starts.append((fx, x))
                
        # If we didn't find 3 distinct ones, fill with best remaining
        if len(selected_starts) < 3:
            for fx, x in eval_results:
                if (fx, x) not in selected_starts:
                    selected_starts.append((fx, x))
                if len(selected_starts) >= 3:
                    break

        # Phase 3: Nelder-Mead Refinement
        # Allocate remaining budget to these 3 agents.
        # Remaining budget: self.budget - calls_made
        # We'll run Nelder-Mead on each.
        # Nelder-Mead can be expensive. We should limit the number of function calls per agent.
        # scipy.optimize.minimize allows options={'maxiter': ...} but doesn't strictly limit function calls in all versions.
        # It's safer to use a custom loop or just trust it with a low maxiter.
        # For dim=5, Nelder-Mead needs ~6 points initially.
        
        remaining_budget = self.budget - calls_made
        budget_per_agent = max(1, remaining_budget // 3)
        
        for i, (f_start, x_start) in enumerate(selected_starts):
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Run Nelder-Mead
            try:
                result = minimize(
                    safe_func_v2, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': 0.5}
                )
            except Exception:
                continue
                
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
                
        return best_x, best_f

# EVOLVE-BLOCK-END
