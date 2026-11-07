"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits linear structure of f5 by estimating gradient and jumping to boundary."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        # Internal function to check deadline
        def timed_out():
            return time.time() > deadline

        # 1. Initial point at center
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # 2. Estimate gradient using central differences with a small step
        # The function is linear (plus noise/offset/shift/rotation which preserve linearity in local sense)
        # f(x) = a^T x + b
        # We want to find a.
        # Use a step size that is large enough to overcome numerical precision but small enough to stay in the "linear" regime (which is global here).
        # However, the box constraint clips. So we must ensure x +/- h * e_i stays in [-5, 5].
        # Center is 0, so h=0.1 is safe.
        h = 0.1
        grad = np.zeros(self.dim)
        
        for i in range(self.dim):
            if timed_out():
                break
                
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            
            # Ensure within bounds (though at 0, +/- 0.1 is safe)
            # But just in case, clip
            x_plus = np.clip(x_plus, -5, 5)
            x_minus = np.clip(x_minus, -5, 5)
            
            f_plus = func(x_plus)
            calls += 1
            f_minus = func(x_minus)
            calls += 1
            
            grad[i] = (f_plus - f_minus) / (2 * h)
            
            if timed_out():
                break

        # 3. Determine optimal corner
        # For f(x) = a^T x + b, minimum on [-5, 5]^d is at x_i = -5 if a_i > 0, else 5.
        # If a_i is very small, it doesn't matter much, but we follow the sign.
        x_opt = np.zeros(self.dim)
        for i in range(self.dim):
            if grad[i] > 0:
                x_opt[i] = -5.0
            else:
                x_opt[i] = 5.0
        
        f_opt_est = func(x_opt)
        calls += 1
        
        if f_opt_est < best_fx:
            best_x = x_opt.copy()
            best_fx = f_opt_est

        # 4. If we still have budget and time, perform a local refinement around x_opt
        # It's possible the gradient estimate was slightly off due to non-linearities (though f5 is linear)
        # or numerical issues. A simple hill climbing or Nelder-Mead from x_opt can polish.
        # Given the linear nature, x_opt should be very close.
        # Let's do a few steps of (1+1)-ES starting from x_opt with small sigma to be safe.
        
        # If we have more than 50 calls left, we can do a bit of local search.
        # But actually, for a linear function, the corner IS the optimum. 
        # Unless the "hidden shift" moves the minimizer outside the box? 
        # "minimizer lies in [-4, 4]^5". So the minimizer is INSIDE the box.
        # Wait. "minimizer lies in [-4, 4]^5".
        # If the function is linear f(x) = a^T(x - x*) + f*, the minimum is at x*.
        # If x* is inside [-5, 5]^5, then the gradient points towards x*.
        # The minimum on the box [-5, 5]^5 would be x* if x* is inside.
        # But my logic above assumes the minimum is on the boundary.
        # If the minimizer is inside, the gradient is zero at x*? No, for a linear function, the gradient is constant.
        # A linear function has no local minimum inside the domain unless the gradient is zero.
        # If the gradient is zero, f is constant.
        # If the gradient is non-zero, the minimum is on the boundary.
        # BUT the task says "minimizer lies in [-4, 4]^5".
        # This implies that the *global* minimum of the unconstrained function is inside the box?
        # Or does it mean the *constrained* minimizer is inside?
        # "The minimizer lies in [-4, 4]^5".
        # If f is linear, the unconstrained minimum is at infinity unless grad=0.
        # Therefore, the "minimizer" referred to must be the *constrained* minimizer?
        # Or is the function not purely linear?
        # "BBOB-style function f5 (Linear slope)".
        # In BBOB, f5 is "Linear slope".
        # Definition: f(x) = sum(x_i) + 10 * sum(x_i^2) ... no that's different.
        # Let's check BBOB f5 definition.
        # BBOB f5: Linear slope.
        # f(x) = sum_{i=1}^d x_i + 10 * sum_{i=1}^d x_i^2 ? No.
        # Actually, BBOB functions are often transformed.
        # "Linear slope" usually means f(x) = a^T x + b.
        # If f(x) = a^T x + b, the minimum on the box is at a corner.
        # Why does the task say "minimizer lies in [-4, 4]^5"?
        # Maybe the "minimizer" is the point where the *unconstrained* function would be minimized if it had a quadratic term?
        # Or maybe the "shift" makes the center of the quadratic term (if it exists) lie there?
        # Let's look at the inspiration: "Exploiting the linear structure of f5 by estimating the constant gradient and jumping directly to the optimal boundary corner... achieving the minimum function value in ~6 evaluations".
        # This suggests the previous successful program jumped to a boundary corner and it worked.
        # This implies the effective function on the box behaves like a linear function where the minimum is at a boundary.
        # If the minimizer were strictly inside, a linear function would be flat (grad=0), which is a degenerate case.
        # It is more likely that the "minimizer lies in [-4, 4]^5" refers to the *location of the minimum of the underlying quadratic* if there is one, or it's a constraint on where the *optimal solution* is found, which might contradict the "boundary corner" finding if the function is strictly linear.
        # However, if the function is strictly linear, the minimum MUST be on the boundary.
        # Perhaps the "minimizer" text is a red herring or refers to a different property (e.g. the shift vector is in that range).
        # "Instances apply a hidden seeded shift, rotation and offset".
        # If there is a shift, the function might be f(R(x-s) + o).
        # If the base function is linear, the transformed function is linear.
        # So the minimum is at a corner of the box in the *transformed* space?
        # No, the box is fixed in the input space.
        # If f(x) is linear, min is at a corner.
        # Let's trust the "boundary corner" strategy from the inspiration which scored 2.575.
        
        # To be robust, let's do a quick local search from the estimated corner.
        # If the corner is correct, local search will confirm or slightly improve.
        # If the corner is wrong (e.g. due to noise), local search can recover.
        
        sigma = 0.5
        x_curr = best_x.copy()
        f_curr = best_fx
        
        # Simple (1+1)-ES for remaining budget
        while calls < self.budget and not timed_out():
            y = np.clip(x_curr + sigma * np.random.default_rng(calls).standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < f_curr:
                x_curr, f_curr = y, fy
                if f_curr < best_fx:
                    best_x, best_fx = x_curr.copy(), f_curr
                sigma *= 1.5
            else:
                sigma *= 0.87
                # If sigma becomes too small, we are converged
                if sigma < 1e-6:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
