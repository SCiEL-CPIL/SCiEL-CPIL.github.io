"""Gradient-Adaptive Search for BBOB Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using a gradient-adaptive step size strategy.
    
    Strategy:
    1. Estimate the local gradient magnitude using central finite differences.
    2. Scale the step size inversely to the gradient magnitude (or curvature proxy).
    3. Use a deterministic "gradient descent" style move but with adaptive step size 
       to balance exploration and exploitation.
    4. Since we are black-box and the function is Sphere (convex, smooth), 
       gradient information is highly effective.
       
    The key structural change from Simulated Annealing is the use of 
    directional information (gradient) to guide the step, rather than 
    random directions accepted based on a temperature schedule.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        best_f = fx
        best_x = x.copy()
        
        # Parameters for adaptive step size
        # We estimate the gradient magnitude to determine how "steep" the landscape is.
        # If the gradient is large, we are far from the optimum (for Sphere, grad ~ 2*(x-x_opt)).
        # We want steps that are proportional to the distance to the optimum.
        
        # Base step size for finite difference estimation
        # This should be small enough to be in the local linear/quadratic region 
        # but large enough to be distinguishable from numerical noise (though we have exact floats).
        # For Sphere f(x) = ||x-x_opt||^2 + offset, the gradient is linear.
        # A fixed small h is fine for estimation.
        h = 1e-3 
        
        # Initial step size for the main move
        # Start with a moderately large step to cover the box quickly.
        step_size = 4.0
        
        # Adaptation parameters
        # We will adjust step_size based on whether the move improved the function.
        # However, the directive suggests "adaptively scaled by the inverse of the local gradient magnitude".
        # Let's implement a scheme where the step size is roughly ||x - x_opt|| / ||grad|| * constant.
        # Since we don't know x_opt, we approximate the "distance to optimum" by the step size that 
        # would zero out the gradient if it were linear.
        
        # For Sphere, grad = 2(x - x_opt). So x_opt = x - grad/2.
        # The ideal step is -grad / (2 * lambda) where lambda is the Hessian eigenvalue (which is 2 for standard sphere).
        # So ideal step = -grad / 4.
        # Let's use this insight.
        
        while calls < budget and time.time() < deadline:
            # 1. Estimate Gradient
            # Use central differences for better accuracy
            grad = np.zeros(dim)
            for i in range(dim):
                if calls + 2 > budget or time.time() > deadline:
                    break
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                
                # Clip to box
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls += 2
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            if calls >= budget or time.time() > deadline:
                break
                
            # 2. Compute Gradient Magnitude
            grad_norm = np.linalg.norm(grad)
            
            # 3. Determine Step Direction and Size
            # If gradient is very small, we are close to the optimum.
            # Use a smaller step to avoid overshooting or wasting calls on tiny improvements.
            # If gradient is large, take a larger step.
            
            # The "ideal" step for a quadratic with constant Hessian H is -H^{-1} grad.
            # For Sphere, H = 2I (assuming standard scaling). So step = -grad / 2.
            # However, the scaling of the BBOB function might vary.
            # Let's estimate the local curvature scale.
            # If we assume f ~ c * ||x - x_opt||^2, then grad = 2c(x - x_opt).
            # Step to optimum is -grad / (2c).
            # We don't know c. But we can adapt.
            
            # Strategy:
            # Use the gradient direction.
            # Scale the step size by the gradient magnitude, but capped.
            # If grad_norm is small, use a minimum step size to explore locally.
            # If grad_norm is large, use a step proportional to grad_norm.
            
            # A robust heuristic for Sphere:
            # step = -grad / (2 * estimated_curvature)
            # Let's estimate curvature from the change in gradient? No, too expensive.
            
            # Alternative: Use the ratio of current function value to gradient magnitude?
            # For Sphere: f = ||d||^2, grad = 2d. ||grad|| = 2||d||.
            # f / ||grad|| = ||d||^2 / (2||d||) = ||d|| / 2.
            # So distance to optimum is proportional to f / ||grad||.
            
            if grad_norm > 1e-6:
                # Estimate distance to optimum
                dist_est = fx / grad_norm
                # The step should be in the direction of -grad, with magnitude roughly dist_est.
                # But we need to be careful if fx is offset. 
                # f(x) = ||x-x_opt||^2 + offset.
                # grad = 2(x-x_opt).
                # f(x) - offset = ||x-x_opt||^2.
                # We don't know offset.
                # However, the direction is correct.
                # Let's use a step size that is a fraction of the estimated distance.
                # To be safe, let's use a fixed fraction of the gradient norm, 
                # but scaled by an adaptation factor.
                
                # Let's try: step_vec = -grad * (step_size / grad_norm)
                # This makes the step magnitude equal to step_size.
                # We adapt step_size based on progress.
                
                # Initial step_size was 4.0.
                # If we move in -grad direction, we should improve.
                # Let's calculate the proposed move.
                direction = -grad / grad_norm
                x_new = x + direction * step_size
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = func(x_new)
                calls += 1
                
                # 4. Adapt Step Size
                if fx_new < fx:
                    # Improvement. We can increase step size to explore further, 
                    # or keep it if we are converging fast.
                    # For Sphere, if we step exactly to the optimum, grad becomes 0.
                    # If we overshoot, we might still improve but less.
                    # Let's increase step size slightly to cover the box faster initially.
                    step_size *= 1.1
                    if step_size > 8.0:
                        step_size = 8.0
                    
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # No improvement. We overshot or the step was too large.
                    # Reduce step size.
                    step_size *= 0.5
                    if step_size < 1e-4:
                        # If step size is tiny and no improvement, we are likely at the optimum 
                        # or stuck in numerical noise. 
                        # Try a random small jitter to escape potential flat spots (though Sphere has none).
                        step_size = 1e-4
                        
            else:
                # Gradient is zero. We are at the stationary point.
                # For Sphere, this is the global minimum.
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
