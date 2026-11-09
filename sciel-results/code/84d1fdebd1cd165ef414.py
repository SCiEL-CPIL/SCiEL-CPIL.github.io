import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        mid = (lo + hi) / 2.0
        ranges = hi - lo
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Assume first 3 vars are shape, rest are distance
        n_shape = 3
        n_tail = dim - n_shape
        
        x_shape_base = mid[:n_shape]
        x_tail_base = mid[n_shape:]
        
        evaluated_X = []
        evaluated_F = []
        
        def eval_point(x):
            nonlocal budget
            if time.time() > deadline:
                return None
            f = problem(x)
            evaluated_X.append(x)
            evaluated_F.append(f)
            budget -= 1
            return f
            
        # Phase 1: Find good tail
        # Use 10 evals to find a tail vector that minimizes the objectives
        # Combine with base shape
        best_tail = x_tail_base.copy()
        best_sum = np.inf
        
        for _ in range(10):
            if budget <= 0 or time.time() > deadline:
                break
            x_tail_rand = lo[n_shape:] + (hi[n_shape:] - lo[n_shape:]) * np.random.rand(n_tail)
            x = np.concatenate([x_shape_base, x_tail_rand])
            f = eval_point(x)
            if f is None: continue
            s = np.sum(f)
            if s < best_sum:
                best_sum = s
                best_tail = x_tail_rand
        
        # Phase 2: Grid search on shape variables
        # 5x5x5 grid = 125 points
        grid_size = 5
        grid_points = []
        # Create grid
        l_shape = lo[:n_shape]
        h_shape = hi[:n_shape]
        
        # Generate grid
        g1 = np.linspace(l_shape[0], h_shape[0], grid_size)
        g2 = np.linspace(l_shape[1], h_shape[1], grid_size)
        g3 = np.linspace(l_shape[2], h_shape[2], grid_size)
        
        for v1 in g1:
            for v2 in g2:
                for v3 in g3:
                    if budget <= 0 or time.time() > deadline:
                        break
                    x_shape = np.array([v1, v2, v3])
                    x = np.concatenate([x_shape, best_tail])
                    eval_point(x)
        
        # Phase 3: Local refinement
        # Get non-dominated points from evaluated_F
        if evaluated_F:
            F = np.array(evaluated_F)
            X = np.array(evaluated_X)
            
            # Simple non-dominated sort
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            dominated = np.zeros(n, dtype=bool)
            
            # Find front 0
            front0 = []
            for i in range(n):
                is_dom = False
                for j in range(n):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    front0.append(i)
                    
            # If front0 is empty, pick best by sum
            if not front0:
                front0 = [np.argmin(np.sum(F, axis=1))]
            
            # Select top 10 from front0 (or all if less than 10)
            candidates = front0[:10]
            
            # Refine each candidate with 6 perturbations
            for idx in candidates:
                if budget <= 0 or time.time() > deadline:
                    break
                x_center = X[idx]
                for _ in range(6):
                    if budget <= 0 or time.time() > deadline:
                        break
                    # Gaussian perturbation
                    noise = np.random.randn(dim) * 0.1 * ranges
                    x_new = x_center + noise
                    x_new = np.clip(x_new, lo, hi)
                    eval_point(x_new)
            
            # Use remaining budget for random global search or further refinement
            # Let's do random global search to avoid local optima
            while budget > 0 and time.time() < deadline:
                x_rand = lo + (hi - lo) * np.random.rand(dim)
                eval_point(x_rand)
                
        return np.array(evaluated_X) if evaluated_X else np.array([mid])
