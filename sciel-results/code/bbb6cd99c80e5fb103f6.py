import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7: x[0], x[1] are position variables. x[2:] are distance variables.
        # We want to explore x[0], x[1] and optimize x[2:] for minimal f3.
        
        pop_size = 10
        X = np.zeros((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        
        # Initialize
        for i in range(pop_size):
            # Random position on the front
            X[i, 0] = rng.random()
            X[i, 1] = rng.random()
            # Initialize tail variables to 0.5
            X[i, 2:] = 0.5
            
        for i in range(pop_size):
            F[i] = problem(X[i])
        used = pop_size
        
        # Local search to minimize f3 for each point
        # We have budget - pop_size evaluations left
        remaining_budget = self.budget - used
        steps_per_point = remaining_budget // pop_size
        
        for i in range(pop_size):
            for _ in range(steps_per_point):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Perturb tail variables
                x = X[i].copy()
                f = F[i]
                
                # Random direction in tail space
                dir_tail = rng.standard_normal(self.dim - 2)
                step_size = 0.1
                new_tail = x[2:] + dir_tail * step_size
                new_tail = np.clip(new_tail, lo[2:], hi[2:])
                
                new_x = x.copy()
                new_x[2:] = new_tail
                
                new_f = problem(new_x)
                used += 1
                
                # Accept if f3 improves (since f1, f2 are fixed)
                # Actually, we should check if it dominates or just improves f3
                # Since f1, f2 are unchanged, improving f3 is always good for hypervolume
                # unless it makes f1 or f2 worse? No, f1=x1, f2=x2 are fixed.
                if new_f[2] < f[2]:
                    X[i] = new_x
                    F[i] = new_f
            
        # Use remaining budget to refine or add more points?
        # With 200 budget, 10 points * 19 steps = 190. 10 left.
        # Let's do a few global mutations on the best points.
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Pick a random point to mutate
            idx = rng.integers(pop_size)
            x = X[idx].copy()
            f = F[idx]
            
            # Perturb position variables slightly
            x[0] += rng.normal(0, 0.05)
            x[1] += rng.normal(0, 0.05)
            x[0] = np.clip(x[0], lo[0], hi[0])
            x[1] = np.clip(x[1], lo[1], hi[1])
            
            # Re-optimize tail quickly (1 step)
            dir_tail = rng.standard_normal(self.dim - 2)
            x[2:] += dir_tail * 0.05
            x[2:] = np.clip(x[2:], lo[2:], hi[2:])
            
            new_f = problem(x)
            used += 1
            
            # Replace if it improves f3 (simple heuristic)
            # Or if it dominates?
            # Since we want hypervolume, keeping points with low f3 is key.
            if new_f[2] < f[2]:
                X[idx] = x
                F[idx] = new_f
                
        return X
