"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope) using Exact Linear Model Identification.
    
    Strategy:
    1. The function is effectively linear: f(x) = g^T x + b.
    2. We identify the linear model exactly by evaluating at d+2 points:
       - Origin x=0 gives b = f(0).
       - Points x_i = 5*e_i for i=1..d give g_i = (f(5*e_i) - f(0)) / 5.
    3. Once we have g, we know the gradient direction.
    4. For a linear function on a box [-5, 5]^d, the minimum is at a vertex 
       where x_i = -5 if g_i > 0, else x_i = 5.
    5. We jump directly to this optimal vertex.
    6. This uses d+2 = 7 calls. The first call (origin) gives a decent initial 
       estimate, and the 7th call (vertex) gives the exact optimum.
    7. Early AOCC is improved because we find the exact optimum very quickly.
       After call 7, the error is ~0, so n_i ~ 0 for the rest of the budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        def timed_out():
            return time.time() > deadline

        dim = self.dim
        budget = self.budget
        calls = 0
        
        # 1. Evaluate at origin to get b and initial best
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        best_x = x0.copy()
        best_f = f0
        
        # 2. Evaluate at 5*e_i for each dimension to get gradient components
        g = np.zeros(dim)
        for i in range(dim):
            if timed_out() or calls >= budget:
                break
                
            xi = np.zeros(dim)
            xi[i] = 5.0
            fi = func(xi)
            calls += 1
            
            # g_i = (f(5*e_i) - f(0)) / 5
            g[i] = (fi - f0) / 5.0
            
            # Update best if this probe point is better (unlikely but safe)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        # 3. Construct the optimal vertex
        # If g_i > 0, we want x_i as small as possible -> -5
        # If g_i < 0, we want x_i as large as possible -> 5
        x_opt = np.where(g > 0, -5.0, 5.0)
        
        # 4. Evaluate the optimal vertex
        if calls < budget and not timed_out():
            f_opt = func(x_opt)
            calls += 1
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # 5. If we have significant budget left, we can do a verification or 
        #    handle potential non-linearity, but for f5 it's linear.
        #    We can also just return immediately since we've found the optimum.
        #    However, the contract says "More than 500 calls... makes the whole 
        #    result invalid". We are well within budget.
        #    We can stop here. The remaining calls are not needed.
        
        return best_x, best_f

# EVOLVE-BLOCK-END
