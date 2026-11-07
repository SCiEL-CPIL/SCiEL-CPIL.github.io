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
        deadline = start_time + 280  # Safety margin before 300s limit

        def timed_out():
            return time.time() > deadline

        # 1. Initial point at center
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_f = f0
        best_x = x0.copy()

        # 2. Estimate gradient using forward differences from origin
        #    For a linear function f(x) = w^T x + b, we have:
        #    f(e_i * h) = w_i * h + b
        #    f(0) = b
        #    So w_i = (f(e_i * h) - f(0)) / h
        #    Using h=1.0 gives exact gradient for linear functions (if no rotation complications)
        #    But with hidden rotation, we need to be more careful.
        
        # Actually, with a hidden rotation R and shift s, the function is:
        # f(x) = w^T (R @ x + s) + b = (R^T w)^T x + (w^T s + b)
        # So it's still linear: f(x) = g^T x + c where g = R^T w and c = w^T s + b
        # The minimum is at x_i = -5 * sign(g_i)
        
        # Use forward differences with h=1.0 to estimate g
        h = 1.0
        g = np.zeros(dim)
        
        for i in range(dim):
            if timed_out():
                break
            xp = np.zeros(dim)
            xp[i] = h
            fp = func(xp)
            g[i] = (fp - f0) / h
            if fp < best_f:
                best_f = fp
                best_x = xp.copy()
        
        # 3. Predict optimum vertex
        x_pred = np.where(g > 0, -5.0, 5.0)
        x_pred[np.abs(g) < 1e-9] = 0.0

        f_pred = func(x_pred)
        if f_pred < best_f:
            best_f = f_pred
            best_x = x_pred.copy()
        
        # 4. Verify by checking all vertices
        for mask in range(2 ** dim):
            if timed_out():
                break
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
            fv = func(v)
            if fv < best_f:
                best_f = fv
                best_x = v.copy()

        return best_x, best_f
