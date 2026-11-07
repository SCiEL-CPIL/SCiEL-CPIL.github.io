import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290 # 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameterization: x1 and S (sum of x2..x10)
        # x = [x1, S/9, S/9, ..., S/9]
        
        def get_x(x1, S):
            x = np.zeros(self.dim)
            x[0] = x1
            x[1:] = S / (self.dim - 1)
            # Clip to bounds
            x = np.clip(x, lo, hi)
            return x

        def evaluate(x1, S):
            if time.time() > deadline:
                return None
            x = get_x(x1, S)
            f = problem(x)
            return (x1, S, x, f)

        records = [] # list of (x1, S, x, f)
        
        # Phase 1: Grid Search
        # x1 in [0, 1], S in [0, 1]
        # 10x10 grid = 100 evals
        n_grid = 10
        x1_vals = np.linspace(0, 1, n_grid)
        S_vals = np.linspace(0, 1, n_grid)
        
        for x1 in x1_vals:
            for S in S_vals:
                if len(records) >= self.budget: break
                res = evaluate(x1, S)
                if res:
                    records.append(res)
                    
        if time.time() > deadline:
            return [r[2] for r in records]

        # Phase 2: Greedy Refinement
        # We want to maximize hypervolume.
        # Approximation: Keep non-dominated set.
        # We will perform local searches around non-dominated points.
        
        def is_dominated(f, records_list):
            # Check if f is dominated by any point in records_list
            for _, _, _, f_other in records_list:
                if np.all(f_other <= f) and np.any(f_other < f):
                    return True
            return False

        def get_non_dominated(records_list):
            nd = []
            for r in records_list:
                if not is_dominated(r[3], records_list):
                    nd.append(r)
            return nd

        # Initial non-dominated set
        nd_set = get_non_dominated(records)
        
        evals_left = self.budget - len(records)
        
        # Refinement loop
        while evals_left > 0 and time.time() < deadline:
            if not nd_set:
                break
                
            # Pick a random non-dominated point
            idx = np.random.randint(len(nd_set))
            parent = nd_set[idx]
            p_x1, p_S, _, _ = parent
            
            # Generate neighbors
            # Random perturbation
            n_neighbors = min(5, evals_left)
            for _ in range(n_neighbors):
                if time.time() > deadline: break
                if evals_left <= 0: break
                
                # Perturb x1 and S
                dx1 = np.random.normal(0, 0.1)
                dS = np.random.normal(0, 0.1)
                
                n_x1 = p_x1 + dx1
                n_S = p_S + dS
                
                # Keep within bounds
                if n_x1 < 0 or n_x1 > 1: continue
                if n_S < 0 or n_S > 1: continue # S max is roughly 1 for f1<11? 
                # Actually S can go up to 9, but f1 = 1 + 10*S/9.
                # If S=1, f1=2. If S=9, f1=11.
                # Reference is 11. So S can go up to ~0.9.
                # Let's allow S up to 1.0 for safety.
                
                res = evaluate(n_x1, n_S)
                if res:
                    records.append(res)
                    evals_left -= 1
                    # Check if new point is non-dominated
                    if not is_dominated(res[3], records):
                        nd_set.append(res)
                        # Optionally remove dominated points from nd_set? 
                        # Not strictly necessary for generation, but keeps list clean.
                        # For now, just append.
                        
        return [r[2] for r in records]
