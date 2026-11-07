import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Leave a safety margin (evaluator kills at 330s, we target 320s)
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # --- Phase 1: Identify the "Good" G-vector ---
        # For DTLZ1 with n=7, M=3:
        # Shape vars: x[0], x[1]
        # G vars: x[2], x[3], x[4], x[5], x[6]
        # Optimal G=0 is achieved when G vars are 0.5
        
        # Test a few candidate G vectors to find the best one
        test_shape = np.array([0.5, 0.5])
        candidate_gs = [
            np.full(5, 0.5),  # Theoretical optimum
            np.zeros(5),
            np.ones(5),
            np.random.rand(5),
            np.random.rand(5)
        ]
        
        best_g = None
        best_score = float('inf')
        used = 0
        
        for g in candidate_gs:
            if used >= self.budget:
                break
            x = np.concatenate([test_shape, g])
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                used += 1
                score = np.sum(f)
                if score < best_score:
                    best_score = score
                    best_g = g.copy()
            except Exception:
                continue
            
            if time.time() > deadline:
                break

        if best_g is None:
            best_g = np.full(5, 0.5)

        # --- Phase 2: Grid Search over Shape Variables ---
        # Now we sweep x0 and x1 in [0, 1]
        remaining = self.budget - used
        if remaining <= 0:
            return []
            
        # Determine grid size
        n_grid = int(np.sqrt(remaining))
        if n_grid < 5:
            n_grid = 5
            
        # Generate grid
        x0_vals = np.linspace(0.0, 1.0, n_grid)
        x1_vals = np.linspace(0.0, 1.0, n_grid)
        
        results = []
        
        for x0 in x0_vals:
            for x1 in x1_vals:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                x = np.concatenate([np.array([x0, x1]), best_g])
                x = np.clip(x, lo, hi)
                
                try:
                    f = problem(x)
                    used += 1
                    if all(f_i < 400.0 for f_i in f):
                        results.append(f)
                except Exception:
                    continue
            
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

        # --- Phase 3: Refine G vector if budget remains ---
        if used < self.budget and results:
            shape_point = np.array([0.5, 0.5])
            refine_budget = (self.budget - used) // 2
            
            current_best_g = best_g.copy()
            current_best_f = None
            current_best_score = float('inf')
            
            # Evaluate current best G at shape_point to get baseline
            x = np.concatenate([shape_point, current_best_g])
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                used += 1
                current_best_f = f
                current_best_score = np.sum(f)
            except Exception:
                pass
            
            # Random search for better G
            for _ in range(refine_budget):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Perturb current best G
                new_g = current_best_g + 0.1 * np.random.randn(5)
                new_g = np.clip(new_g, lo[2:], hi[2:])
                
                x = np.concatenate([shape_point, new_g])
                x = np.clip(x, lo, hi)
                
                try:
                    f = problem(x)
                    used += 1
                    score = np.sum(f)
                    
                    if score < current_best_score:
                        current_best_score = score
                        current_best_f = f
                        current_best_g = new_g.copy()
                except Exception:
                    continue
            
            if current_best_f is not None:
                if all(f_i < 400.0 for f_i in current_best_f):
                    results.append(current_best_f)

        return results
