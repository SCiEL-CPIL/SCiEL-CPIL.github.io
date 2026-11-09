"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Initial random sampling to estimate the linear model parameters (a, b).
    2. Use the estimated gradient to jump to the best corner of the box.
    3. Evaluate all 2^dim corners to guarantee finding the global minimum of the linear model.
    4. Refine with (1+1)-ES in case of non-linearity or estimation errors.
    5. Deadline guard to ensure we finish within 300s (with margin).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        rng = np.random.default_rng(0)
        
        best_x = None
        best_fx = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= self.budget or time.time() > deadline:
                return None
            calls_made += 1
            x = np.asarray(x, dtype=float)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Estimate gradient using finite differences from a random starting point
        # We use a random point to avoid potential symmetry issues, though for linear functions
        # any point works. We use step size 1.0 for robustness.
        
        x_start = rng.uniform(-5, 5, self.dim)
        f_start = safe_call(x_start)
        
        if f_start is None:
            # Fallback if something went wrong immediately
            return (np.zeros(self.dim), np.inf) if best_x is None else (best_x, best_fx)

        # Estimate gradient components
        eps = 1.0
        a_est = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline or calls_made >= self.budget:
                break
                
            xi = x_start.copy()
            xi[i] += eps
            xi = np.clip(xi, -5, 5)
            
            if np.allclose(xi, x_start):
                # Try negative direction
                xi = x_start.copy()
                xi[i] -= eps
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    continue
                fi = safe_call(xi)
                if fi is None:
                    break
                a_est[i] = (fi - f_start) / (-eps)
            else:
                fi = safe_call(xi)
                if fi is None:
                    break
                a_est[i] = (fi - f_start) / eps

        # Phase 2: Evaluate all corners of the box [-5, 5]^dim
        # For a linear function f(x) = a^T x + b, the minimum over the box is at a corner.
        # There are 2^dim corners. For dim=5, that's 32 corners.
        # We have used 1 (start) + 5 (gradient) = 6 calls.
        # We can afford to evaluate all 32 corners.
        
        num_corners = 2 ** self.dim
        for mask in range(num_corners):
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(self.dim)])
            safe_call(v)
        
        # If best_x is still None (should not happen), initialize
        if best_x is None:
            best_x = x_start.copy()
            best_fx = f_start

        # Phase 3: Refine with (1+1)-ES
        # The function might not be perfectly linear due to the "hidden seeded shift, rotation and offset"
        # or if my interpretation of "Linear slope" is slightly off (e.g., if it's actually a cone or has a kink).
        # However, BBOB f5 is typically linear. But to be safe, we do a local search.
        # Start from the best corner found.
        
        x_cur = best_x.copy()
        fx_cur = best_fx
        sigma = 0.5  # Start with a moderate step size
        
        # Adaptive sigma for (1+1)-ES
        # We use the remaining budget for refinement.
        
        while calls_made < self.budget and time.time() < deadline:
            y = x_cur + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = safe_call(y)
            if fy is None:
                break
            
            if fy < fx_cur:
                x_cur = y
                fx_cur = fy
                sigma *= 1.5  # Increase step size on success
            else:
                sigma *= 0.85  # Decrease step size on failure
            
            # If sigma becomes too small, we might be stuck or converged
            if sigma < 1e-6:
                break

        return best_x, best_fx

# EVOLVE-BLOCK-END
