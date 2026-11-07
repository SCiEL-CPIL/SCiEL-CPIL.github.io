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
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        best_x = x0.copy()
        best_fx = f0
        
        # Check time
        if time.time() > deadline:
            return best_x, best_fx
            
        # Step 2: Evaluate at 5 * e_i for each dimension to estimate gradient
        # f(5*e_i) - f(0) = 5 * w_i
        w_est = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                break
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            w_est[i] = (fi - f0) / 5.0
            
            # Update best if this point is better (unlikely but possible if w_i < 0 and we are at 5, wait, min is at -5 if w_i > 0)
            # Actually, we are just estimating. We will evaluate the true vertex later.
            
        # Step 3: Construct optimal vertex based on estimated gradient signs
        # If w_i > 0, min is at -5
        # If w_i < 0, min is at 5
        x_opt = np.zeros(d)
        for i in range(d):
            if w_est[i] > 1e-8:
                x_opt[i] = -5.0
            elif w_est[i] < -1e-8:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0 # Doesn't matter
        
        # Step 4: Evaluate the optimal vertex
        if time.time() > deadline:
            return best_x, best_fx
            
        f_opt = func(x_opt)
        if f_opt < best_fx:
            best_x = x_opt.copy()
            best_fx = f_opt
        else:
            # If the estimated vertex wasn't better, something is wrong (e.g. noise or non-linearity)
            # Fallback: try the other vertex? Or just return best so far.
            # Given the linear assumption, this shouldn't happen unless w_i is extremely close to 0.
            # We can try to verify by checking a random vertex or just returning.
            # To be safe, let's check the opposite vertex if the improvement was marginal or negative?
            # No, for linear function, the vertex IS the minimum.
            # If f_opt >= best_fx, it means the estimated vertex is not better than 0.
            # This implies the estimated signs were wrong for some components?
            # Or w is very small.
            # In any case, best_fx is the best we have.
            pass

        # Optional: If we have budget and time, we could verify by checking a few other vertices
        # But with 7 calls we are already very good.
        
        return best_x, best_fx
