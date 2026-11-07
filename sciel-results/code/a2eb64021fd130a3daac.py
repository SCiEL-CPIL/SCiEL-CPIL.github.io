"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear Slope).
    Strategy:
    1. Evaluate the center (0,0,0,0,0).
    2. Evaluate points along each axis (4.0 * e_i) to estimate the gradient direction.
    3. Determine the optimal vertex based on the gradient signs.
    4. Evaluate that vertex.
    5. Return the best found.
    This finds the global minimum of a linear function in 7 calls.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                calls_made += 1
                return func(x)
            except Exception:
                return None

        # 1. Evaluate center
        x0 = np.zeros(self.dim)
        f0 = safe_call(x0)
        if f0 is not None:
            best_f = f0
            best_x = x0.copy()
        else:
            # If center fails, we can't proceed effectively with gradient estimation
            # Try a random point?
            x_rand = np.random.default_rng(0).uniform(-5, 5, self.dim)
            f_rand = safe_call(x_rand)
            if f_rand is not None:
                best_f = f_rand
                best_x = x_rand.copy()
            return best_x, best_f

        # 2. Evaluate axis points to estimate gradient
        # Use h=4.0 for a strong signal while staying within [-5, 5]
        h = 4.0
        grad_signs = []
        
        for i in range(self.dim):
            xi = np.zeros(self.dim)
            xi[i] = h
            fi = safe_call(xi)
            
            if fi is None:
                # If call fails, assume gradient is 0 or unknown. 
                # Default to 0 (no preference) or keep previous best.
                # For robustness, if we can't determine sign, we might miss the optimum.
                # But for linear functions, this shouldn't happen.
                grad_signs.append(0) 
                continue
                
            # If fi > f0, the function increases in the +i direction.
            # So the minimum is in the -i direction (i.e., x_i = -5).
            # If fi < f0, the function decreases in the +i direction.
            # So the minimum is in the +i direction (i.e., x_i = +5).
            if fi > f0:
                grad_signs.append(-1)
            else:
                grad_signs.append(1)

        # 3. Construct candidate vertex
        x_star = np.array([5.0 * s for s in grad_signs])
        f_star = safe_call(x_star)
        
        if f_star is not None and f_star < best_f:
            best_f = f_star
            best_x = x_star.copy()

        # 4. Optional: If we have budget and time, we could verify with the opposite vertex?
        # No, for linear functions, the vertex determined by signs is the unique global minimum (unless gradient is 0 in some dims).
        # If gradient is 0 in a dim, both -5 and 5 are equally good.
        
        # We have used at most 7 calls. We can return now.
        # The evaluator will repeat the last n_i for the remaining budget.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
