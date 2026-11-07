"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    Strategy:
    1. Estimate the gradient using central differences (2D+1 calls).
    2. Jump to the predicted optimal vertex of the box [-5, 5]^D.
    3. Use remaining budget for a short local polish (Nelder-Mead) to handle 
       any non-linearities or precision issues, though the function is linear.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 10 seconds
        deadline = start_time + 290.0
        
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if time.time() > deadline:
                return float('inf')
            calls_made += 1
            return func(x)

        # 1. Gradient Estimation
        # We use a step size of 1.0 for central differences.
        # This requires 1 + 2*D calls.
        x0 = np.zeros(self.dim)
        f0 = safe_call(x0)
        
        if time.time() > deadline:
            return x0, f0

        g = np.zeros(self.dim)
        delta = 1.0
        
        for i in range(self.dim):
            xp = np.zeros(self.dim)
            xm = np.zeros(self.dim)
            xp[i] = delta
            xm[i] = -delta
            
            fp = safe_call(xp)
            fm = safe_call(xm)
            
            if time.time() > deadline:
                break
            
            g[i] = (fp - fm) / (2.0 * delta)

        # 2. Determine Optimal Corner
        # For a linear function f(x) = g^T x + c:
        # If g[i] > 0, f decreases as x[i] decreases -> min at -5
        # If g[i] < 0, f decreases as x[i] increases -> min at 5
        x_corner = np.zeros(self.dim)
        for i in range(self.dim):
            if g[i] > 1e-12:
                x_corner[i] = -5.0
            elif g[i] < -1e-12:
                x_corner[i] = 5.0
            else:
                x_corner[i] = 0.0 # Gradient is zero, location doesn't matter for this dim

        f_corner = safe_call(x_corner)
        
        best_x = x0
        best_f = f0
        if f_corner < best_f:
            best_x = x_corner
            best_f = f_corner

        # 3. Polish Phase
        # If we have budget left, run a short Nelder-Mead optimization.
        # This is a safeguard in case the function isn't perfectly linear 
        # or to refine if the corner estimate was slightly off due to noise.
        remaining_time = deadline - time.time()
        if remaining_time > 1.0 and calls_made < self.budget:
            # Limit iterations to ensure we finish in time
            # Nelder-Mead is fast. 100 iterations is very quick for dim=5.
            try:
                res = minimize(
                    safe_call, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'fatol': 1e-12, 'xatol': 1e-12}
                )
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
