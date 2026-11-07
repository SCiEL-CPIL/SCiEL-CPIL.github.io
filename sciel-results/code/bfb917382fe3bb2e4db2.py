"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizes the BBOB f5 (Linear slope) function.
    
    The function is linear: f(x) = g^T x + c (after rotation/shift).
    The minimum over [-5,5]^d is at a vertex where each coordinate is
    either -5 or +5 depending on the sign of the corresponding gradient component.
    
    Strategy:
    1. Evaluate at center (0,...,0) to get f(0) = c.
    2. For each dimension i, probe x with x[i]=5 (others 0) to estimate g[i].
       Actually, we need to be careful: the function may have been rotated.
       But the task says "rotation" is applied, so the effective function in
       the original coordinates is still linear: f(x) = a^T x + b for some
       vector a and scalar b.
       
       Wait, let me reconsider. The task says "hidden seeded shift, rotation and offset".
       A rotation of the input space means f(x) = f0(Rx + s) + o where f0 is the
       base linear function. If f0(y) = c^T y + d, then f(x) = c^T (Rx + s) + d + o
       = (R^T c)^T x + (c^T s + d + o). So it's still linear in x.
       
       So we can estimate the gradient by finite differences.
       
    3. Determine the optimal vertex and evaluate it.
    
    For AOCC, we want early progress. So we should:
    - First evaluate center (call 1)
    - Then sequentially determine each coordinate, updating the best as we go.
    
    But actually, the current approach already does this. Let me think about
    how to improve it.
    
    Key insight: For a linear function, once we know the gradient, we know
    the optimum exactly. The issue is that we need to estimate the gradient.
    
    Current approach uses dim+1 evaluations total (center + dim probes).
    But we can do better for AOCC by interleaving:
    
    Alternative: Use a smarter probing strategy that gives partial information
    earlier. For example:
    
    1. Evaluate center: f0 = f(0)
    2. For each dimension i:
       a. Probe +5 in dim i: f_plus_i = f(e_i * 5)
       b. Probe -5 in dim i: f_minus_i = f(-e_i * 5)
       c. grad_i = (f_plus_i - f_minus_i) / 10
       d. Set x[i] = -5 if grad_i > 0 else +5
       e. Evaluate the new partial vertex
    
    But this uses 2*dim + 1 = 11 evaluations, same as before in the worst case.
    
    Actually, the current program uses at most dim+1 = 6 evaluations because
    it reuses the +5 probe when the optimal direction is +5.
    
    Let me think differently. The key for AOCC is to improve the best-so-far
    quickly. 
    
    New strategy:
    1. Evaluate center (call 1): f_best = f(0)
    2. For i in 0..dim-1:
       a. Probe +5 in dim i from current best: f_plus
       b. Probe -5 in dim i from current best: f_minus  
       c. Choose the better one as the new value for dim i
       d. Evaluate the new point (if it's different from what we probed)
    
    Wait, this is 2 probes per dimension = 2*dim = 10 probes + 1 center = 11 calls.
    But we get a valid point at each step.
    
    Actually, let's think more carefully. After probing both +5 and -5 for dim i,
    we know which is better. We then set x[i] to that value. The point we evaluate
    is the one with the better value, which we already have from the probe.
    
    So:
    - Call 1: f(0)
    - Calls 2-3: f(+5 e_0), f(-5 e_0) -> choose better, update x[0]
    - Calls 4-5: f(x with x[1]=+5), f(x with x[1]=-5) -> choose better, update x[1]
    - ...
    
    Total: 1 + 2*dim = 11 calls.
    
    But wait, after setting x[0], the base point for probing dim 1 is no longer
    the origin. It's the point with x[0] set optimally and x[1..4]=0.
    
    This is fine because the function is linear, so the gradient estimate is still
    valid regardless of the base point.
    
    Let me implement this.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 290
        calls = 0

        def safe_call(x):
            nonlocal calls
            if time.time() > deadline:
                return None
            calls += 1
            return func(x)

        # 1. Evaluate center
        x_best = np.zeros(dim)
        f_best = safe_call(x_best)
        if f_best is None:
            return x_best, f_best

        # 2. For each dimension, probe both directions and choose the better
        for i in range(dim):
            if time.time() > deadline:
                break

            # Probe +5
            x_plus = x_best.copy()
            x_plus[i] = 5.0
            f_plus = safe_call(x_plus)
            if f_plus is None:
                break

            # Probe -5
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
