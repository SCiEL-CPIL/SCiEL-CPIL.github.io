"""Rewritten program: Gradient-based corner finder for linear BBOB f5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer for linear slope function: find optimal corner via gradient estimation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Step 1: Estimate gradient by evaluating center and axis perturbations
        center = np.zeros(d)
        fc = func(center)
        calls_used = 1

        # Evaluate positive and negative axis points to determine gradient signs
        # For linear f(x) = a^T x + b, f(center + h*e_i) - f(center) = h * a_i
        h = 4.0  # step size to stay in box
        grad_signs = np.zeros(d)
        
        for i in range(d):
            if calls_used >= budget:
                break
            pos = center.copy()
            pos[i] = h
            fpos = func(pos)
            calls_used += 1
            
            neg = center.copy()
            neg[i] = -h
            fneg = func(neg)
            calls_used += 1
            
            # If fpos < fneg, gradient component is negative (minimize by going positive)
            # If fneg < fpos, gradient component is positive (minimize by going negative)
            if fpos < fneg:
                grad_signs[i] = 1.0  # go positive
            else:
                grad_signs[i] = -1.0  # go negative

        # The optimal corner for linear function is determined by gradient signs
        # For f(x) = a^T x + b, min is at x_i = -5 * sign(a_i)
        # Wait, let me reconsider: if a_i > 0, we want x_i as small as possible (-5)
        # if a_i < 0, we want x_i as large as possible (5)
        # So optimal corner: x_i = -5 if a_i > 0, else 5
        # From our test: if fpos < fneg, then a_i * h < -a_i * h, so a_i < 0
        # If a_i < 0, optimal x_i = 5 (positive)
        # If a_i > 0, optimal x_i = -5 (negative)
        
        optimal_corner = np.zeros(d)
        for i in range(d):
            if grad_signs[i] == 1.0:
                # fpos < fneg means a_i < 0, so go to +5
                optimal_corner[i] = 5.0
            else:
                # a_i > 0, so go to -5
                optimal_corner[i] = -5.0

        f_corner = func(optimal_corner)
        calls_used += 1

        # Now we have a good candidate. Let's also try the opposite corner
        # and a few other corners to make sure
        best_x = optimal_corner.copy()
        best_f = f_corner

        # Try a few more corners to verify
        # The 2^d corners, but we only have limited budget
        # Let's try the opposite corner
        if calls_used < budget:
            opposite = -optimal_corner
            f_opp = func(opposite)
            calls_used += 1
            if f_opp < best_f:
                best_x = opposite
                best_f = f_opp

        # Now refine: for a linear function, the minimum is exactly at the corner
        # But due to the hidden shift, the true minimum might be slightly inside
        # Let's do a local search around the best corner
        # Actually for pure linear, the corner IS the minimum if it's in the box
        
        # Use remaining budget to do a more careful gradient estimation
        # and potentially find a better point if the function isn't perfectly linear
        # or if my corner identification was wrong
        
        # Let's do a more careful check: evaluate a few points along the 
        # direction from center to best corner
        if calls_used < budget - 10:
            # Binary search or line search from center toward best corner
            # to verify the direction is correct
            alpha_vals = [0.5, 0.75, 0.9, 0.95, 0.99]
            for alpha in alpha_vals:
                if calls_used >= budget:
                    break
                x_test = alpha * best_x
                f_test = func(x_test)
                calls_used += 1
                if f_test < best_f:
                    best_x = x_test
                    best_f = f_test

            # If the function is truly linear, f(alpha * best_x) should be
            # monotonically decreasing as alpha increases toward 1
            # If it's not, we might have the wrong corner
            
            # Try a few more random corners to be safe
            rng = np.random.default_rng(123)
            for _ in range(min(20, budget - calls_used)):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                # Random corner
                corner = rng.choice([-5, 5], size=d)
                f_c = func(corner)
                calls_used += 1
                if f_c < best_f:
                    best_x = corner
                    best_f = f_c

        # Final: use remaining budget to do fine-tuning
        # For a linear function, we're already at the optimum corner
        # But let's do a small local search in case of numerical issues
        if calls_used < budget:
            sigma = 0.1
            for _ in range(budget - calls_used):
                if time.time() > deadline:
                    break
                y = best_x + sigma * np.random.randn(d)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x = y
                    best_f = fy
                    sigma *= 0.9
                else:
                    sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
