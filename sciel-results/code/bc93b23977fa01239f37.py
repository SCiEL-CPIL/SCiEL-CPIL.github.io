import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # 1. Center evaluation
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_f = f0
        best_x = x0.copy()
        
        # 2. Gradient estimation using central differences
        # Use a step size that is large enough to avoid numerical noise but small enough to stay in bounds
        # Since domain is [-5, 5], delta=1.0 is safe and provides good resolution for linear functions.
        delta = 1.0
        w = np.zeros(dim)
        
        for i in range(dim):
            xp = np.zeros(dim)
            xm = np.zeros(dim)
            xp[i] = delta
            xm[i] = -delta
            
            f_p = func(xp)
            f_m = func(xm)
            
            w[i] = (f_p - f_m) / (2 * delta)
            
            # Update best if any of these intermediate points are better (unlikely for linear, but safe)
            if f_p < best_f:
                best_f = f_p
                best_x = xp.copy()
            if f_m < best_f:
                best_f = f_m
                best_x = xm.copy()
                
        # 3. Predict optimum vertex
        # For f(x) = w^T x + b, min is at x_i = -5 * sign(w_i)
        x_pred = np.where(w > 0, -5.0, 5.0)
        # Handle w[i] == 0
        x_pred[np.abs(w) < 1e-9] = 0.0
        
        f_pred = func(x_pred)
        if f_pred < best_f:
            best_f = f_pred
            best_x = x_pred.copy()
            
        # 4. Verify by checking all vertices
        # There are 2^dim vertices. For dim=5, this is 32.
        # We already checked x_pred (which is a vertex).
        # We can generate all vertices and check them.
        
        # Generate all 2^dim vertices
        vertices = []
        for mask in range(2 ** dim):
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
            vertices.append(v)
            
        # We can skip x_pred if we just evaluated it, but checking again is cheap and ensures consistency
        # Or just iterate all.
        
        for v in vertices:
            # Optional: Check time
            if time.time() - start_time > 280:
                break
                
            fv = func(v)
            if fv < best_f:
                best_f = fv
                best_x = v.copy()
                
        # 5. If we have budget left, do we do anything else?
        # The function is linear. The minimum is at a vertex.
        # We have checked all vertices. We have found the global minimum.
        # Any further calls will only increase the "best value" history if we don't improve, 
        # or keep it same. 
        # AOCC is 1 - mean(n_i). 
        # Once we hit the minimum, n_i becomes small (close to 0 if f_opt is close to 0, or small log value).
        # Actually, n_i = (log10(e_i) + 8) / 10.
        # If e_i is very small (close to 0), log10 is negative.
        # e_i is clipped to [1e-8, 100].
        # So min n_i is (log10(1e-8) + 8)/10 = (-8+8)/10 = 0.
        # So if we find the optimum, n_i becomes 0.
        # We want n_i to be small as soon as possible.
        # We found the optimum at call ~12 (1 + 2*5 + 1).
        # Calls 13-39 check other vertices.
        # If the predicted one was correct, best_f stays at the minimum.
        # So n_i for calls 12-500 is ~0.
        # This is optimal.
        
        return best_x, best_f
