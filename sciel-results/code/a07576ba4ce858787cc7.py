"""Model-based optimizer for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Phase 1: Use a set of well-distributed points to estimate the linear model
       f(x) ≈ c + g^T x. Since f5 is a linear slope function (after rotation/shift),
       the gradient g points away from the minimizer. We want to minimize, so we
       move in direction -g.
       
    2. The function is linear: f(x) = a + b^T (x - x_opt) where x_opt is the minimizer
       and b is the slope vector. Since it's linear, the minimum over the box [-5,5]^d
       is at a corner, but the problem says the minimizer lies in [-4,4]^5.
       
    Actually, for BBOB f5 "Linear slope", the function is:
    f(x) = 100 + 0.05 * |x - x_opt|_1  ... no wait, that's not linear slope.
    
    Let me reconsider. BBOB f5 is actually "Linear slope" which is:
    f(x) = 100 + 0.05 * sum_{i=1}^{d} x_i  (after some transformation)
    
    Actually the standard BBOB f5 is: f(x) = 100 + 0.05 * ||x||_1  or similar.
    But the task says "Linear slope" and the minimizer is in [-4,4]^5.
    
    For a linear function f(x) = c + g^T x, the minimum over a box is at a corner.
    But the problem states the minimizer lies in [-4,4]^5, which suggests it might
    not be purely linear, or the "minimizer" refers to where the function achieves
    its minimum within the feasible region.
    
    Wait - re-reading: "the minimizer lies in [-4, 4]^5". For a truly linear function,
    the minimum over [-5,5]^5 would be at a corner (one of ±5 in each dimension).
    Unless the linear function has very small slope in some directions, making the
    minimum effectively at an interior point... but that doesn't make sense for linear.
    
    Perhaps f5 in this context is not purely linear but has a linear component plus
    some other structure. Or perhaps the "minimizer" is approximate.
    
    Let me just use a robust approach:
    1. Sample points to estimate gradient
    2. Use gradient descent with adaptive step size
    3. Fall back to (1+1)-ES for refinement
    
    Key insight: For a linear function, we can determine the direction of steepest
    descent from just a few samples, then take a large step towards the boundary
    in that direction.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            calls_made += 1
            return func(np.asarray(x, dtype=float))
        
        # Phase 1: Estimate gradient using central differences at origin
        # Sample at origin and along each axis
        center = np.zeros(self.dim)
        f0 = safe_call(center)
        if f0 is None:
            return center, np.inf
        
        best_x = center.copy()
        best_f = f0
        
        # Check deadline
        if time.time() > deadline:
            return best_x, best_f
        
        # Estimate gradient components
        step = 1.0
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            if time.time() > deadline or calls_made >= self.budget:
                break
            xi = center.copy()
            xi[i] += step
            fi = safe_call(xi)
            if fi is None:
                break
            grad[i] = (fi - f0) / step
        
        # For a linear function, grad points in direction of increase.
        # We want to go in direction -grad to minimize.
        # The minimizer over [-5,5]^d for a linear function is at the corner
        # where x_i = -5 * sign(grad_i) for each i (if grad_i != 0).
        
        # But the problem says minimizer is in [-4,4]^5, so maybe it's not purely linear.
        # Let's do a more careful approach:
        
        # Phase 2: Use the gradient estimate to take a large step
        # Project -grad onto the box
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            direction = -grad / grad_norm
            # Find the maximum step we can take before hitting the boundary
            # x_new = x + t * direction, keep in [-5, 5]
            t_max = np.inf
            for i in range(self.dim):
                if direction[i] > 1e-15:
                    t_max = min(t_max, (5 - center[i]) / direction[i])
                elif direction[i] < -1e-15:
                    t_max = min(t_max, (-5 - center[i]) / direction[i])
            t_max = max(t_max, 0)
            
            # Try several step sizes along this direction
            for t_frac in [1.0, 0.8, 0.6, 0.4, 0.2]:
                if time.time() > deadline or calls_made >= self.budget:
                    break
                x_try = center + t_frac * t_max * direction
                x_try = np.clip(x_try, -5, 5)
                f_try = safe_call(x_try)
                if f_try is None:
                    break
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
        
        # Phase 3: Refine with (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        
        while calls_made < self.budget and time.time() < deadline:
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = safe_call(y)
            if fy is None:
                break
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.3
            else:
                sigma *= 0.85
            
            # Adaptive sigma floor
            if sigma < 1e-6:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
