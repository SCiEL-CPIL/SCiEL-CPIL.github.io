"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Estimate the linear model parameters (gradient 'a' and offset 'b') 
       using finite differences from the origin with a large step (h=5).
       This uses 1 + dim = 6 calls to get an exact gradient estimate
       for a linear function.
    2. Construct the optimal vertex x* directly from the sign of the gradient.
       For a linear function f(x) = a^T x + b on [-5, 5]^d, the minimum is
       at x_i = -5 * sign(a_i).
    3. Evaluate x* at call 7. This is the global minimum.
    4. Return the result.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin
        calls_made = 0
        dim = self.dim
        budget = self.budget

        # 1. Start at origin to estimate gradient cleanly
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_fx = f0

        # 2. Estimate gradient using forward differences with large step
        # f(x) = a^T x + b. f(0) = b.
        # f(h e_i) = a_i h + b.
        # a_i = (f(h e_i) - f(0)) / h
        # Using h=5 puts the points at the box boundary.
        h = 5.0
        a_est = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            
            xi = x0.copy()
            xi[i] += h
            fi = func(xi)
            calls_made += 1
            
            a_est[i] = (fi - f0) / h
            
            if fi < best_fx:
                best_fx = fi
                best_x = xi.copy()

        # 3. Construct the optimal vertex directly
        # For f(x) = a^T x + b, min is at x_i = -5 if a_i > 0, else 5.
        # If a_i is 0, the function is flat in that dimension. We pick 0.
        x_star = np.zeros(dim)
        for i in range(dim):
            if a_est[i] > 0:
                x_star[i] = -5.0
            elif a_est[i] < 0:
                x_star[i] = 5.0
            else:
                x_star[i] = 0.0

        # 4. Evaluate the optimal vertex
        if calls_made < budget:
            f_star = func(x_star)
            calls_made += 1
            
            if f_star < best_fx:
                best_fx = f_star
                best_x = x_star.copy()
        
        # 5. If budget remains, we can perform a simple local search to handle
        # any potential non-linearity or numerical noise, although for a pure
        # linear function this is unnecessary.
        if calls_made < budget and time.time() < deadline:
            sigma = 0.5
            x_cur = best_x.copy()
            f_cur = best_fx
            no_improve_count = 0
            max_no_improve = 20
            
            while calls_made < budget and time.time() < deadline:
                if no_improve_count > max_no_improve:
                    break
                
                step = np.random.randn(dim) * sigma
                x_new = x_cur + step
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = func(x_new)
                calls_made += 1
                
                if f_new < f_cur:
                    f_cur = f_new
                    x_cur = x_new.copy()
                    best_fx = f_new
                    best_x = x_new.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # Decay sigma if stuck
                    sigma *= 0.9

        return best_x, best_fx

# EVOLVE-BLOCK-END
