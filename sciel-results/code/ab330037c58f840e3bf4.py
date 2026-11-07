"""Rewritten program using Model-and-Optimize approach for Linear Slope."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer exploiting the linear structure of f5 (Linear slope).
    Estimates the gradient and performs large steps towards the minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Hard kill limit is 330s. We set a safety margin of 40s.
        deadline = start_time + 290.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # 1. Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        # 2. Estimate Gradient
        # We estimate the gradient at the current best point x.
        # Since f is linear, the gradient is constant everywhere.
        # We use a forward difference with a small step h.
        # To be robust against noise (if any, though f5 is deterministic), 
        # we could average, but let's stick to a simple estimate first.
        
        # We need to spend some budget on estimation.
        # Let's use 2 * dim calls for a central difference estimate or 
        # dim calls for forward difference.
        # Forward difference: f(x + h*e_i) - f(x) / h
        # Let's use h = 0.01
        
        h = 0.01
        grad = np.zeros(dim)
        
        # Check if we have enough budget for estimation
        if calls + dim < budget:
            for i in range(dim):
                if time.time() > deadline:
                    break
                x_test = x.copy()
                x_test[i] += h
                # Clip to box to ensure valid evaluation, though gradient 
                # estimation near boundaries might be slightly biased if 
                # the function is not linear *outside* the box, but the 
                # problem states f5 is linear slope, usually defined on 
                # the whole space or clipped. The task says "func(x) clips...".
                # So if x+h is out of bounds, the clip changes the effective 
                # gradient. However, the minimizer is in [-4,4]^5, and we 
                # start at 0. h=0.01 is safe.
                fy_test = func(x_test)
                calls += 1
                grad[i] = (fy_test - fx) / h
                
                # Early termination if time is critical
                if time.time() > deadline:
                    break
        else:
            # Not enough budget for full gradient estimation, fall back to random walk
            # But 500 budget is plenty for 5 dim.
            pass

        # 3. Use Gradient to predict Minimum
        # f(x) = f(0) + grad^T x (approx)
        # We want to minimize f.
        # The minimum of a linear function on a box is at one of the corners 
        # defined by the sign of the gradient.
        # If grad[i] > 0, we want x[i] to be as small as possible (-5).
        # If grad[i] < 0, we want x[i] to be as large as possible (5).
        
        # However, the function is rotated. The "linear slope" in BBOB f5 
        # is typically f(x) = sum(a_i * x_i) + b.
        # The rotation is applied to the input. 
        # So the effective gradient in the input space is what we estimated.
        
        # Predicted optimal point based on linear model:
        x_pred = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                x_pred[i] = -5.0
            else:
                x_pred[i] = 5.0
        
        # Evaluate the predicted point
        if calls < budget:
            fx_pred = func(x_pred)
            calls += 1
            if fx_pred < best_fx:
                best_fx = fx_pred
                best_x = x_pred.copy()
            else:
                # If the prediction didn't improve, maybe the gradient estimate 
                # was noisy or the linear assumption is slightly off due to 
                # clipping effects? Or maybe we are already at the minimum?
                # Let's try a point closer to the current best, moving against 
                # the gradient.
                pass

        # 4. Refinement Phase
        # Now that we have a good candidate, use (1+1)-ES to polish.
        # Start from best_x.
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5
        
        # Adapt sigma based on the gradient magnitude? 
        # If gradient is large, we can take bigger steps.
        # But we are likely near the boundary or the minimum is at the boundary.
        # Let's just do a standard (1+1)-ES with adaptive sigma.
        
        success_counter = 0
        total_steps = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate neighbor
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()
                sigma *= 1.2
                success_counter += 1
            else:
                sigma *= 0.9
                success_counter = 0
            
            total_steps += 1
            
            # Simple restart logic if stuck
            if total_steps > 50 and success_counter == 0:
                # Restart from best_x with larger sigma
                x = best_x.copy()
                fx = best_fx
                sigma = 1.0
                total_steps = 0
                success_counter = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
