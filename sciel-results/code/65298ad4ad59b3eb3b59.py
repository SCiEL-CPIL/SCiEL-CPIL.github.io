"""Combined program: Exact linear optimization with aggressive early convergence tracking."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Evaluate at the origin (0,0,0,0,0) to get the base value.
    2. Probe each axis at the positive boundary (+5) to estimate the gradient direction.
       - If f(+5e_i) < f(0), the gradient is negative in that dimension, so the minimum
         is likely at +5 for that dimension.
       - If f(+5e_i) > f(0), the gradient is positive, so the minimum is likely at -5.
       - Note: Probing only positive axes is sufficient to determine the sign of the gradient.
         We don't need to probe -5 separately for the sign, as f(+5e_i) - f(0) = 5*c_i.
         If c_i > 0, min is at -5. If c_i < 0, min is at +5.
    3. Construct the candidate optimal vertex based on the signs.
    4. Evaluate the candidate optimal vertex.
    5. Track the best value found at every single function call to maximize AOCC.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        d = self.dim
        best_x = np.zeros(d)
        best_f = float('inf')
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Evaluate at the origin
        # This establishes the baseline.
        x0 = np.zeros(d)
        f0 = func(x0)
        update_best(x0, f0)
        
        # 2. Probe each axis at the positive boundary (+5)
        # This gives us the gradient component c_i = (f(5e_i) - f(0)) / 5
        # We also update the best here, because if c_i is negative, 
        # f(5e_i) will be significantly lower than f(0), improving the curve early.
        c = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                return best_x, best_f
            
            probe = np.zeros(d)
            probe[i] = 5.0
            f_probe = func(probe)
            
            # Update best immediately to capture early improvements
            update_best(probe, f_probe)
            
            # Calculate gradient component
            c[i] = (f_probe - f0) / 5.0

        # 3. Construct the optimal vertex
        # If c[i] > 0, the function increases in +i direction, so min is at -5.
        # If c[i] < 0, the function decreases in +i direction, so min is at +5.
        x_opt = np.zeros(d)
        for i in range(d):
            if c[i] > 0:
                x_opt[i] = -5.0
            elif c[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0 # Neutral, stays at 0

        # 4. Evaluate the optimal vertex
        if time.time() > deadline:
            return best_x, best_f
            
        f_opt = func(x_opt)
        update_best(x_opt, f_opt)
        
        # 5. The remaining budget (493 calls) will not improve the value further 
        # for a linear function, but the evaluator requires us to finish.
        # We do not need to make dummy calls; the evaluator handles the "repeating the last n_i"
        # if fewer calls are made. However, to be safe and adhere to standard optimizer behavior,
        # we simply return. The contract says "More than 500 calls... makes the whole result invalid".
        # It does not require exactly 500. "repeating the last n_i if fewer calls are made"
        # implies < 500 is fine.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
