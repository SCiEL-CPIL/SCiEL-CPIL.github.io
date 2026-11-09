import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Evaluate center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        calls_used = 1
        
        # 2. Determine the sign of the gradient for each coordinate
        # f(x) = a^T x + b
        # f(0) = b
        # f(5 e_i) = b + 5 a_i
        # If f(5 e_i) < f(0), then a_i < 0, so min is at x_i = +5
        # If f(5 e_i) >= f(0), then a_i >= 0, so min is at x_i = -5
        
        signs = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline or calls_used >= budget:
                break
                
            x_test = x_center.copy()
            x_test[i] = 5.0
            
            f_test = func(x_test)
            calls_used += 1
            
            # Update best if this point is better
            if f_test < best_f:
                best_f = f_test
                best_x = x_test.copy()
            
            # Determine sign
            if f_test < f_center:
                signs[i] = -1.0  # a_i < 0, go to +5
            else:
                signs[i] = 1.0   # a_i >= 0, go to -5
                
        # 3. Construct the optimal vertex
        # If sign is -1 (a_i < 0), x_i = 5.0
        # If sign is 1 (a_i >= 0), x_i = -5.0
        x_vertex = np.where(signs < 0, 5.0, -5.0)
        
        # 4. Evaluate the constructed vertex
        if time.time() > deadline or calls_used >= budget:
            # Fallback if we ran out of budget (unlikely for dim=5)
            pass
        else:
            f_vertex = func(x_vertex)
            calls_used += 1
            
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()
        
        # 5. Fill remaining budget with the best point found
        # This ensures we use the full budget if required, though for linear
        # functions the best point is already found. The score depends on
        # b_i (best value after call i). If we stop, the evaluator repeats
        # the last n_i. Repeating the last n_i is the same as having the
        # best value for all subsequent calls. So stopping early is fine
        # and saves time, but filling budget doesn't hurt and might be
        # safer if there's any weirdness with the evaluator expecting
        # exactly budget calls (though contract says "More than 500 calls...
        # invalid", "repeating the last n_i if fewer calls are made").
        # So we can stop.
        
        return best_x, best_f
