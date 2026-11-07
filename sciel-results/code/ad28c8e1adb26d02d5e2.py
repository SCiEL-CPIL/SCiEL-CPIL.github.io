"""
Strategy: Adaptive Sequential Model Refinement (ASMR) for Linear Slope (f5)
Instead of local search, we exploit the global linearity.
1. Sample 2 points to initialize a linear model f(x) = c + g^T x.
2. Predict the corner that minimizes this model.
3. Evaluate the prediction.
4. If the prediction is close to the model's predicted minimum, we have likely found the global minimum (or very close).
5. If not, the model was inaccurate (due to noise or clipping effects), so we update the model using the new point and the best previous point, and repeat.
This minimizes the number of calls needed to reach the optimum, maximizing early AOCC.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        # Helper to update best
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
                
        # Helper to check time
        def time_ok():
            return time.time() < deadline and calls < budget

        # Phase 1: Initial Model Construction
        # We need at least 2 points to estimate a linear function in high dims? 
        # Actually, for a linear function f(x) = c + g^T x, we need to estimate c and g.
        # A robust way is to use the origin and a point far away in a random direction, 
        # or better, use points that span the space.
        
        # Start with 0
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        update_best(x0, f0)
        
        if not time_ok():
            return best_x, best_fx

        # To estimate gradient g, we can use finite differences from x0.
        # But since it's linear, any two points x_a, x_b give:
        # f(x_b) - f(x_a) = g^T (x_b - x_a)
        # If we pick x_b = x_a + d, then g^T d = f(x_b) - f(x_a).
        # To get the full gradient, we need dim independent directions.
        # However, we can be smarter. 
        # Let's estimate g using a set of points.
        # Actually, a simpler approach for linear functions:
        # The minimum is at a corner. 
        # If we can estimate the sign of each component of g, we can jump to the corner.
        # g_i > 0 => x_i should be -5
        # g_i < 0 => x_i should be 5
        
        # Let's estimate g_i by probing along each axis from x0.
        # This is O(dim) calls. With dim=5, that's 5 calls.
        # Total 6 calls to get a good gradient estimate.
        # Then 1 call to the predicted corner. Total 7 calls.
        
        # Let's try to be even more efficient if possible, but 7 is already very low.
        # The key is accuracy. If the gradient estimate is wrong, we jump to the wrong corner.
        # Using a larger step size h improves robustness against noise (if any) but 
        # clipping might interfere. The problem states f5 is linear slope, so it's likely 
        # strictly linear within the box, or clipped. "clips a finite length-5 vector to the box".
        # This implies the function is defined on the box, and inputs are clipped.
        # So f(x) is linear on [-5,5]^5.
        
        h = 1.0  # Large step to avoid numerical issues and maximize signal
        grad = np.zeros(dim)
        
        for i in range(dim):
            if not time_ok():
                break
            x_probe = x0.copy()
            x_probe[i] += h
            # Clip to ensure it stays in bounds for the probe logic, though x0=0 and h=1 is safe
            x_probe = np.clip(x_probe, -5, 5)
            f_probe = func(x_probe)
            calls += 1
            update_best(x_probe, f_probe)
            
            # Calculate gradient component
            # f(x0 + h*e_i) - f(x0) = g_i * h
            grad[i] = (f_probe - f0) / h
            
        if not time_ok():
            return best_x, best_fx

        # Predict the corner
        # If grad[i] > 0, f increases with x[i], so min is at x[i] = -5
        # If grad[i] < 0, f decreases with x[i], so min is at x[i] = 5
        # If grad[i] == 0, it doesn't matter, but let's stick to 0 or 5? 
        # Usually "Linear slope" implies non-zero gradient.
        # Let's use the sign.
        x_pred = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                x_pred[i] = -5.0
            else:
                x_pred[i] = 5.0
        
        # Evaluate prediction
        if not time_ok():
            return best_x, best_fx
            
        f_pred = func(x_pred)
        calls += 1
        update_best(x_pred, f_pred)
        
        # Check if we are "close" to the model's minimum
        # The model predicts f(x_pred) should be f0 + grad^T x_pred
        f_model_pred = f0 + np.dot(grad, x_pred)
        
        # If the actual value is close to the model prediction, our model is good.
        # If not, there might be noise or the function isn't perfectly linear (though task says it is).
        # Given the task says "Linear slope", the model should be exact.
        # However, floating point or slight variations might exist.
        
        # If the jump was successful (f_pred < best_fx before jump, or just f_pred is low), 
        # we are likely done. 
        # But to be safe and exploit any remaining budget for AOCC (though AOCC is dominated by early calls),
        # we can try to refine if we have budget.
        
        # Actually, for a linear function, the corner IS the global minimum.
        # If our gradient estimate is correct, x_pred is the global minimum.
        # The only risk is if the gradient estimate was wrong due to noise.
        # But with h=1.0 and linear function, the estimate should be exact.
        
        # Let's verify the gradient estimate by checking another point?
        # Or just trust it.
        
        # If we have budget left, we can do a "sanity check" or local refinement 
        # just in case the function has a small non-linear component or noise.
        # But for f5, it's linear.
        
        # To maximize AOCC, we want the best possible value as early as possible.
        # We have used ~7 calls.
        # If we are at the global minimum, b_i for i>7 will be f_opt.
        # n_i will be 0 (or close to it).
        # AOCC will be very high.
        
        # Let's use the remaining budget to ensure we don't miss anything, 
        # but primarily to satisfy the "call" requirement if we stopped early? 
        # No, the evaluator records every call. 
        # If we make fewer than 500 calls, it repeats the last n_i.
        # So making 500 calls doesn't help if we are stuck at the same value.
        # It only hurts if we make worse calls? No, b_i is the BEST value so far.
        # So b_i is non-increasing.
        # If we are at the global minimum, b_i stays at f_opt.
        # So n_i stays at 0.
        # AOCC = 1 - mean(0) = 1.
        
        # So the goal is to reach f_opt as fast as possible.
        
        # What if the gradient estimate was wrong?
        # Example: Noise in func. 
        # If there is noise, h=1.0 might be too small if noise is large.
        # But "Linear slope" usually implies a clean signal.
        
        # Let's add a refinement phase just in case, using the remaining budget.
        # If we are not at the global minimum, (1+1)-ES might help.
        # But if we are, it will just waste calls (which is fine, as long as we don't exceed budget).
        
        # Actually, if we are at the global minimum, any move will increase f(x).
        # So (1+1)-ES will reject moves and keep x.
        # This is safe.
        
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5
        
        # Use remaining budget for (1+1)-ES
        while time_ok():
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                update_best(x, fx)
                sigma *= 1.1
            else:
                sigma *= 0.9
                if sigma < 1e-5:
                    break

        return best_x, best_fx
