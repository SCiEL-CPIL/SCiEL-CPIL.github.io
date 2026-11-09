# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Evaluate at origin to get baseline f0.
    2. For each dimension i:
       a. Evaluate at 5*e_i to determine gradient component v_i.
       b. Construct a "partial vertex" where dimensions 0..i are set to 
          their optimal values (-5 or 5 based on sign of v_j) and 
          dimensions i+1..d-1 are 0.
       c. Evaluate this partial vertex. This allows the best value b_i 
          to improve incrementally as we determine more dimensions, 
          rather than waiting until all dimensions are known.
    3. After all dimensions are processed, the last partial vertex is 
       the global optimum for the linear function.
    4. Stall with the best point found for the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        
        d = self.dim
        calls_made = 0
        x_best = np.zeros(d)
        f_best = np.inf
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        f_best = f0
        x_best = x0.copy()
        
        # Track gradient components
        v = np.zeros(d)
        v_known = np.zeros(d, dtype=bool)
        
        # Step 2: Interleave axis probes and partial vertex evaluations
        for i in range(d):
            # Check time and budget
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # 2a. Probe axis i at distance 5
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            
            # Update gradient component
            v[i] = (fi - f0) / 5.0
            v_known[i] = True
            
            # Update best if this axis point is better
            if fi < f_best:
                f_best = fi
                x_best = xi.copy()
            
            # 2b. Construct partial optimal vertex for dimensions 0..i
            # For j <= i, we know v[j]. Set x[j] to -5 if v[j]>0, 5 if v[j]<0.
            # For j > i, x[j] = 0.
            x_partial = np.zeros(d)
            for j in range(i + 1):
                if v[j] > 0:
                    x_partial[j] = -5.0
                else:
                    x_partial[j] = 5.0
            
            # Check time and budget
            if time.time() > deadline or calls_made >= self.budget:
                break
                
            # Evaluate partial vertex
            fp = func(x_partial)
            calls_made += 1
            
            if fp < f_best:
                f_best = fp
                x_best = x_partial.copy()
        
        # Step 3: If we have all components, ensure we've evaluated the full vertex
        # The last partial vertex in the loop (i=d-1) IS the full vertex,
        # but if we broke early, we might not have it. Also, if the loop
        # completed, the last evaluated x_partial is the full vertex.
        # However, if we broke out of the loop due to budget/time before evaluating
        # the last partial vertex, x_best might be an axis point or an earlier partial.
        # Let's check if we have full info and if we can afford one more call
        # to ensure the true optimum is recorded.
        
        if np.all(v_known):
            x_vertex = np.zeros(d)
            for i in range(d):
                if v[i] > 0:
                    x_vertex[i] = -5.0
                else:
                    x_vertex[i] = 5.0
            
            # Only evaluate if we haven't already (calls_made < budget) 
            # and if it's likely we didn't evaluate it in the loop (e.g. broke early)
            # Actually, the loop evaluates x_partial for each i. If i reaches d-1,
            # x_partial is the full vertex. 
            # If we broke early, we might not have evaluated the full vertex.
            # Let's just try to evaluate it if we have budget, as it's the guaranteed optimum.
            if time.time() < deadline and calls_made < self.budget:
                fv = func(x_vertex)
                calls_made += 1
                if fv < f_best:
                    f_best = fv
                    x_best = x_vertex.copy()
        
        # Step 4: Stall with the best point found
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(x_best)
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
