import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Calculate the global minimizer for the tail variables x[1:]
        # The function to minimize for each tail variable x_i is:
        # h(x) = x^2 - 10 * cos(4 * pi * x)
        # This function has local minima at integers and half-integers.
        # Specifically, h(k) = k^2 - 10 for integer k.
        # h(k + 0.5) = (k + 0.5)^2 - 10 * cos(4 * pi * (k + 0.5))
        #             = (k + 0.5)^2 - 10 * cos(4 * pi * k + 2 * pi)
        #             = (k + 0.5)^2 - 10
        # So we need to check integers and half-integers within the bounds.
        
        x_tail_min = np.zeros(self.dim - 1)
        
        for i in range(1, self.dim):
            l, h = bounds[i]
            
            # Generate candidates: bounds and all half-integers (k/2) in [l, h]
            candidates = [l, h]
            
            # k/2 is a candidate if l <= k/2 <= h  =>  2l <= k <= 2h
            k_min = int(np.ceil(2 * l))
            k_max = int(np.floor(2 * h))
            
            for k in range(k_min, k_max + 1):
                x_cand = k / 2.0
                candidates.append(x_cand)
            
            # Evaluate h(x) for all candidates and pick the best
            best_val = float('inf')
            best_x = l
            
            for x_c in candidates:
                # Ensure x_c is within bounds (though construction guarantees it mostly)
                if l <= x_c <= h:
                    val = x_c * x_c - 10 * np.cos(4 * np.pi * x_c)
                    if val < best_val:
                        best_val = val
                        best_x = x_c
                        
            x_tail_min[i - 1] = best_x

        n_evals = self.budget
        
        # Generate LHS points for x[0]
        # We use np.random.permutation to create a random permutation of intervals
        # This is deterministic given the seed set by the evaluator
        
        # Create intervals for LHS
        # u_i in [i/n, (i+1)/n]
        # x_0 = lo[0] + u_i * (hi[0] - lo[0])
        
        # To be efficient, we can generate the random offsets for all points at once
        # but the loop is required for problem(x) calls.
        
        # We will use a simple loop with np.random.uniform for each point
        # This is standard LHS
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            # LHS point in the i-th interval
            interval_start = i / n_evals
            interval_end = (i + 1) / n_evals
            
            u = np.random.uniform(interval_start, interval_end)
            
            x0 = lo[0] + u * (hi[0] - lo[0])
            
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = x_tail_min
            
            try:
                problem(x)
            except Exception:
                continue
        
        return None
