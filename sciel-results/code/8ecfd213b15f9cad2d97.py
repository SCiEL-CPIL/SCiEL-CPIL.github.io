# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using grid-based decomposition with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin against 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Track all evaluations
        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget or time.time() > deadline:
                return None
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                if not np.all(np.isfinite(f)):
                    f = np.full(n_obj, 1e6)
                all_X.append(x.copy())
                all_F.append(f)
                eval_count += 1
                return f
            except Exception:
                all_X.append(x.copy())
                all_F.append(np.full(n_obj, 1e6))
                eval_count += 1
                return None

        # Phase 1: Coarse grid evaluation
        # Use a 3x3x3x3 grid = 81 points for coarse coverage
        grid_size = 3
        num_grid = grid_size ** dim
        
        # Reserve budget: use ~40% for grid, rest for refinement
        grid_budget = min(num_grid, budget // 2)
        if grid_budget < 16:
            grid_size = 2
            num_grid = grid_size ** dim
            grid_budget = min(num_grid, budget // 2)
        
        # Generate grid points
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * i / (grid_size - 1),
                            lo[1] + (hi[1] - lo[1]) * j / (grid_size - 1),
                            lo[2] + (hi[2] - lo[2]) * k / (grid_size - 1),
                            lo[3] + (hi[3] - lo[3]) * l / (grid_size - 1)
                        ])
                        grid_points.append(x)
        
        # Evaluate grid points
        grid_F = []
        for x in grid_points:
            f = evaluate(x)
            if f is None:
                break
            grid_F.append(f)
        
        if not grid_F:
            return None
        
        grid_F = np.array(grid_F)
        
        # Phase 2: Identify promising regions
        # Find non-dominated points in the grid
        def non_dominated_indices(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            return np.where(~dominated)[0]
        
        nd_idx = non_dominated_indices(grid_F)
        
        # Get the non-dominated grid points
        nd_X = [grid_points[i] for i in nd_idx]
        nd_F = grid_F[nd_idx]
        
        # Phase 3: Local refinement using Nelder-Mead
        # For each non-dominated point, try to improve by minimizing a weighted sum
        # Use multiple weight vectors to explore different parts of the Pareto front
        
        # Define weight vectors for 3 objectives
        weight_vectors = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
            [0.33, 0.33, 0.34],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.4, 0.4],
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
            [0.5, 0.2, 0.3],
            [0.3, 0.2, 0.5],
            [0.2, 0.5, 0.3],
        ]
        
        # Shuffle starting points for diversity
        np.random.shuffle(nd_X)
        
        # Limit number of local optimizations based on remaining budget
        remaining_budget = budget - eval_count
        # Each Nelder-Mead call uses roughly 10-20 evaluations
        max_local_ops = min(len(nd_X), remaining_budget // 15)
        
        for start_idx in range(min(max_local_ops, len(nd_X))):
            if eval_count >= budget or time.time() > deadline:
                break
            
            x0 = nd_X[start_idx]
            
            # Try a few weight vectors from this starting point
            for w in weight_vectors[:5]:
                if eval_count >= budget or time.time() > deadline:
                    break
                
                def objective(x):
                    if eval_count >= budget or time.time() > deadline:
                        return 1e6
                    f = evaluate(x)
                    if f is None:
                        return 1e6
                    return np.dot(w, f)
                
                # Use Nelder-Mead with tight bounds
                bounds_nm = list(zip(lo, hi))
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='Nelder-Mead',
                        bounds=bounds_nm,
                        options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                except Exception:
                    pass
        
        # Phase 4: Additional random sampling if budget remains
        remaining = budget - eval_count
        if remaining > 20 and time.time() < deadline:
            # Add some random points for exploration
            n_random = min(remaining, 30)
            for _ in range(n_random):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.rand(dim)
                evaluate(x)
        
        return None
# EVOLVE-BLOCK-END
