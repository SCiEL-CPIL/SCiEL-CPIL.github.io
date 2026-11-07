import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 325
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Initial random sampling
        # Use a moderate population size to find initial fronts
        init_size = 20
        X = lo + np.random.rand(init_size, self.dim) * span
        F = np.array([problem(x) for x in X])
        used = init_size
        
        # Maintain a list of non-dominated points
        # We will store indices into a larger array or just keep the best points
        # Since we need to record ALL evaluations, we just call problem() and let the evaluator record.
        # We maintain our own internal list of (x, f) for non-dominated points to guide search.
        
        nd_set = [] # List of [x, f]
        
        def is_dominated(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
            
        def update_nd_set(x, f):
            nonlocal nd_set
            # Check if f is dominated by any in nd_set
            for pt in nd_set:
                if is_dominated(f, pt[1]):
                    return False # Dominated, do not add
            # Remove points dominated by f
            new_nd_set = []
            for pt in nd_set:
                if not is_dominated(pt[1], f):
                    new_nd_set.append(pt)
            new_nd_set.append([x, f])
            nd_set = new_nd_set
            return True
            
        # Initialize nd_set
        for i in range(init_size):
            update_nd_set(X[i], F[i])
            
        # Local search parameters
        # We will use a simple perturbation strategy
        # If we stagnate, we jump.
        
        stagnation_counter = 0
        max_stagnation = 10 # If 10 evals don't improve nd_set, jump
        
        while used < self.budget:
            if time.time() > limit_time:
                break
            
            if stagnation_counter >= max_stagnation:
                # Jump: pick a random point
                x_new = lo + np.random.rand(self.dim) * span
                f_new = problem(x_new)
                used += 1
                if update_nd_set(x_new, f_new):
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                # Reset stagnation counter on jump to give it a chance
                # Actually, if the jump point is dominated, we should still reset the counter 
                # to allow further jumps or local searches, otherwise we might get stuck 
                # if we keep picking bad random points. 
                # Let's reset to 0 on any jump to encourage exploration.
                stagnation_counter = 0 
            else:
                # Local search: pick a parent from nd_set
                if len(nd_set) == 0:
                     # Should not happen if init works, but just in case
                     x_new = lo + np.random.rand(self.dim) * span
                else:
                    # Select parent: Random or Diversity-based?
                    # Random is simpler and effective with jumps.
                    # Let's pick a random non-dominated point.
                    parent_idx = np.random.randint(0, len(nd_set))
                    parent_x = nd_set[parent_idx][0]
                    
                    # Perturb
                    # Use Gaussian mutation with adaptive sigma?
                    # Simple: uniform perturbation in a small box
                    # Or polynomial mutation
                    # Let's use a simple Gaussian with sigma = 0.1 * span
                    # But bounds are [0, 1] typically for WFG? No, bounds are given.
                    # Let's use a fraction of the span.
                    sigma = 0.1 * span
                    noise = np.random.normal(0, sigma)
                    x_new = parent_x + noise
                    x_new = np.clip(x_new, lo, hi)
                    
                f_new = problem(x_new)
                used += 1
                
                if update_nd_set(x_new, f_new):
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
        
        # Return the last X or something, doesn't matter
        return X
