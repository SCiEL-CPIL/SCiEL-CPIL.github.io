# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    
    Strategy:
    1. Coarse grid evaluation to systematically cover the 4D domain
    2. Identify non-dominated grid points
    3. Local refinement via Nelder-Mead on weighted sums around promising points
    4. Time guard to ensure completion within 300s
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # Track all evaluated points and their objective values
        all_points = []
        all_objectives = []
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            eval_count += 1
            all_points.append(x)
            all_objectives.append(f)
            return f
        
        # Phase 1: Coarse grid - 3 points per dimension = 81 points
        grid_size = 3
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
        for x in grid_points:
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break
            evaluate(x)
        
        if not all_objectives:
            # Fallback: just do random evaluations
            for _ in range(min(self.budget, 50)):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x)
            return all_points
        
        F = np.array(all_objectives)
        
        # Find non-dominated points from grid
        def is_dominated(f, F_all):
            # Check if f is dominated by any point in F_all
            for other in F_all:
                if np.all(other <= f) and np.any(other < f):
                    return True
            return False
        
        non_dominated_indices = []
        for i, f in enumerate(F):
            if not is_dominated(f, F):
                non_dominated_indices.append(i)
        
        # Phase 2: Local refinement around non-dominated points
        # Use weighted sum approach with Nelder-Mead
        from scipy.optimize import minimize
        
        # Generate weight vectors
        n_weights = 10
        weights = []
        for a in np.linspace(0.1, 0.9, 3):
            for b in np.linspace(0.1, 0.9, 3):
                c = 1.0 - a - b
                if c > 0.05:
                    weights.append([a, b, c])
        
        # Select top non-dominated points for refinement
        max_refinements = min(len(non_dominated_indices), 8)
        selected_indices = non_dominated_indices[:max_refinements]
        
        for idx in selected_indices:
            if time.time() > deadline:
                break
            if eval_count >= self.budget - 5:
                break
            
            center = all_points[idx]
            
            # Try a few weight vectors for local search
            for w in weights[:3]:
                if eval_count >= self.budget - 2:
                    break
                if time.time() > deadline:
                    break
                
                def objective(x):
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    eval_count += 1
                    all_points.append(x)
                    all_objectives.append(f)
                    return np.dot(w, f)
                
                # Nelder-Mead with tight bounds
                try:
                    result = minimize(
                        objective,
                        center,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 15, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
                    )
                except Exception:
                    pass
        
        # Phase 3: If we have budget left, add some random points
        remaining = self.budget - eval_count
        if remaining > 0:
            n_random = min(remaining, 20)
            for _ in range(n_random):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x)
        
        return all_points
# EVOLVE-BLOCK-END
