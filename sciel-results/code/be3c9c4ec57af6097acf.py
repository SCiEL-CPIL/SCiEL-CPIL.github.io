"""Optimizer for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizes the BBOB f5 (Linear slope) function.
    
    The function is linear: f(x) = a^T x + b for some vector a and scalar b.
    The minimum over [-5,5]^d is at a vertex where each coordinate is
    either -5 or +5 depending on the sign of the corresponding gradient component.
    
    Strategy for maximizing AOCC:
    1. Evaluate center (call 1) to get baseline.
    2. Sequentially probe each dimension: for dim i, evaluate x with x[i]=+5 and x[i]=-5
       (keeping previously determined optimal values for dims < i, and 0 for dims > i).
       Choose the better one and update.
    3. This gives us a valid improving point at every step (calls 2, 3, 4, 5, ...).
    
    Actually, a better strategy:
    1. Call 1: f(0) -> baseline
    2. For i in 0..4:
       - Call: f(point with x[i]=+5, others 0 except previously set)
       - Call: f(point with x[i]=-5, others 0 except previously set)
       - Choose better, update x[i]
    
    This uses 1 + 2*5 = 11 calls total, and we get an improving point at calls 3, 5, 7, 9, 11.
    
    Alternative: Use the fact that for a linear function, we can determine the full
    gradient with just dim+1 evaluations (origin + dim axis probes), then jump to the
    optimal vertex. But that only gives one improvement at call 6.
    
    The sequential approach gives intermediate improvements which is better for AOCC.
    
    Let me think about this more carefully:
    - Call 1: f(0) = b
    - Call 2: f(5*e_0) = b + 5*a[0]
    - Call 3: f(-5*e_0) = b - 5*a[0]
      -> We know a[0], set x[0] = -5 if a[0] > 0 else 5
      -> Best so far: min(f(0), f(5*e_0), f(-5*e_0)) = b - 5*|a[0]|
    - Call 4: f(x[0]*e_0 + 5*e_1) = b + x[0]*a[0] + 5*a[1]
    - Call 5: f(x[0]*e_0 - 5*e_1) = b + x[0]*a[0] - 5*a[1]
      -> We know a[1], set x[1] optimally
      -> Best: b - 5*|a[0]| - 5*|a[1]|
    - ...
    - Call 11: optimal vertex, f_opt
    
    So the error decreases at calls 1, 3, 5, 7, 9, 11.
    
    Even better: after we have the gradient for dim i, we can immediately evaluate
    the partial vertex. But we already do that via the probes.
    
    Let me implement the sequential approach.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 280
        calls = 0

        def safe_call(x):
            nonlocal calls
            if time.time() > deadline:
                return None
            calls += 1
            return func(x)

        # Start at origin
        x_best = np.zeros(dim)
        f_best = safe_call(x_best)
        if f_best is None:
            return x_best, f_best

        # Sequentially determine optimal direction for each dimension
        for i in range(dim):
            if time.time() > deadline:
                break

            # Probe +5 in dimension i (keeping previously set values)
            x_plus = x_best.copy()
            x_plus[i] = 5.0
            f_plus = safe_call(x_plus)
            if f_plus is None:
                break

            # Probe -5 in dimension i
            x_minus = x_best.copy()
            x_minus[i] = -5.0
            f_minus = safe_call(x_minus)
            if f_minus is None:
                break

            # Choose the better direction
            if f_plus <= f_minus:
                x_best = x_plus
                f_best = f_plus
            else:
                x_best = x_minus
                f_best = f_minus

        return x_best, f_best
# EVOLVE-BLOCK-END
