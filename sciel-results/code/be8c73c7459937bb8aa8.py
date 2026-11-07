import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        d = self.dim
        budget = self.budget
        call_count = 0
        
        def time_ok():
            return time.time() < deadline and call_count < budget
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        call_count += 1
        best_x = x0.copy()
        best_fx = f0
        
        if not time_ok():
            return best_x, best_fx
        
        # Step 2: Evaluate at -5*e_i for each dimension to estimate gradient
        # f(-5*e_i) - f(0) = -5 * w_i, so w_i = (f0 - fi) / 5
        # If w_i > 0, min is at -5 (we're already probing there)
        # If w_i < 0, min is at +5
        w_est = np.zeros(d)
        for i in range(d):
            if not time_ok():
                break
            xi = np.zeros(d)
            xi[i] = -5.0
            fi = func(xi)
            call_count += 1
            w_est[i] = (f0 - fi) / 5.0
            
            # Update best if this point is better
            if fi < best_fx:
                best_x = xi.copy()
                best_fx = fi
        
        # Step 3: Construct optimal vertex based on estimated gradient signs
        # If w_i > 0, min is at -5
        # If w_i < 0, min is at +5
        x_opt = np.zeros(d)
        for i in range(d):
            if w_est[i] > 1e-8:
                x_opt[i] = -5.0
            elif w_est[i] < -1e-8:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0
        
        # Step 4: Evaluate the optimal vertex
        if not time_ok():
            return best_x, best_fx
            
        f_opt = func(x_opt)
        call_count += 1
        if f_opt < best_fx:
            best_x = x_opt.copy()
            best_fx = f_opt
        
        return best_x, best_fx
