"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Gradient Descent with Central Differences and Curvature Adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Initialize at center
        x = np.zeros(dim)
        fx = func(x)
        call_count = 1
        
        best_x = x.copy()
        best_fx = fx
        
        # Gradient estimation parameters
        # Central difference epsilon: 1e-5 is a good balance for float64
        eps = 1e-5
        
        # Adaptive step size
        # Initial guess: assume curvature ~1, so alpha ~ 0.5
        alpha = 0.5 
        a_est = 1.0 # Estimated curvature
        
        # Threshold for switching to polish mode
        # If gradient norm is small, we are close to the minimum
        polish_threshold = 1e-3
        
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Estimate gradient using Central Differences
            # Cost: 2 * dim calls
            if call_count + 2 * dim > budget:
                # If we can't afford central diff, try forward diff
                if call_count + dim > budget:
                    break
                
                grad = np.zeros(dim)
                for i in range(dim):
                    x_p = x.copy()
                    x_p[i] += eps
                    f_p = func(x_p)
                    call_count += 1
                    grad[i] = (f_p - fx) / eps
            else:
                grad = np.zeros(dim)
                for i in range(dim):
                    x_p = x.copy()
                    x_p[i] += eps
                    x_m = x.copy()
                    x_m[i] -= eps
                    f_p = func(x_p)
                    f_m = func(x_m)
                    call_count += 2
                    grad[i] = (f_p - f_m) / (2 * eps)
            
            grad_norm_sq = np.dot(grad, grad)
            grad_norm = np.sqrt(grad_norm_sq)
            
            # Check convergence
            if grad_norm < 1e-10:
                break
            
            # Determine step size
            # If we are close to the minimum, use a smaller step or switch to polish
            # Otherwise, use curvature-based step size
            
            # Update curvature estimate if we have a recent step
            # For now, we just use the current alpha. 
            # We will update a_est after taking a step.
            
            # If gradient is small, we are in the basin. 
            # Use a more conservative step to avoid overshooting, or line search.
            if grad_norm < polish_threshold:
                # Polish phase: 1D line search along -grad
                # We have budget for a few evaluations.
                # Let's do a simple ternary search or just take a small step.
                # Given the budget, a few small steps might be better than a complex search.
                # Let's try to estimate the optimal step for this 1D slice.
                # f(x - t * grad) is quadratic in t.
                # We can evaluate f at two points to fit the quadratic.
                
                # Cost so far this iteration: 2*dim (or dim)
                # Remaining budget: budget - call_count
                # Let's spend 2 calls to estimate the 1D curvature.
                
                if call_count + 2 <= budget:
                    # Direction is -grad / ||grad||
                    direction = -grad / grad_norm
                    
                    # Test points: x + t1 * direction, x + t2 * direction
                    # t1, t2 should be small.
                    # Use current alpha as a scale.
                    t1 = alpha * 0.5
                    t2 = alpha * 1.5
                    
                    x1 = x + t1 * direction
                    x2 = x + t2 * direction
                    
                    f1 = func(x1)
                    f2 = func(x2)
                    call_count += 2
                    
                    # Fit quadratic f(t) = a*t^2 + b*t + c
                    # We know f(0) = fx
                    # f(t1) = a*t1^2 + b*t1 + fx
                    # f(t2) = a*t2^2 + b*t2 + fx
                    # Solve for a, b.
                    # a*t1^2 + b*t1 = f1 - fx
                    # a*t2^2 + b*t2 = f2 - fx
                    
                    # Matrix form:
                    # [t1^2 t1] [a]   [f1-fx]
                    # [t2^2 t2] [b] = [f2-fx]
                    
                    det = t1*t2*t2 - t2*t1*t1
                    if abs(det) > 1e-12:
                        a_quad = ((f1-fx)*t2 - (f2-fx)*t1) / det
                        b_quad = ((f2-fx)*t1*t1 - (f1-fx)*t2*t2) / det
                        
                        # Optimal t = -b / (2a)
                        if a_quad > 1e-12:
                            t_opt = -b_quad / (2 * a_quad)
                            # Clamp t_opt to reasonable range
                            t_opt = np.clip(t_opt, 0.0, 10.0 * alpha)
                        else:
                            t_opt = alpha # Fallback
                    else:
                        t_opt = alpha # Fallback
                    
                    x_new = x + t_opt * direction
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_new = func(x_new)
                    call_count += 1
                    
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        
                        # Update a_est based on this 1D fit
                        a_est = a_quad / (grad_norm**2) if a_quad > 1e-12 else a_est
                        # Adjust alpha for next full gradient step if we exit polish
                        alpha = 1.0 / (2.0 * a_est) if a_est > 1e-6 else 0.1
                        alpha = np.clip(alpha, 1e-4, 10.0)
                    else:
                        # If the fit was bad or step didn't improve, shrink alpha
                        alpha *= 0.5
                        if alpha < 1e-6:
                            break
                            
            else:
                # Standard Gradient Descent step
                # Use curvature estimate to set alpha
                # alpha_opt = 1 / (2 * a_est)
                # But we need to be careful if a_est is bad.
                # Let's try the estimated alpha, and if it fails, shrink.
                
                current_alpha = alpha
                accepted = False
                for _ in range(5):
                    x_new = x - current_alpha * grad
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_new = func(x_new)
                    call_count += 1
                    
                    if f_new < fx:
                        accepted = True
                        
                        # Update a_est
                        delta_f = f_new - fx
                        # For quadratic: Delta f = -alpha * ||g||^2 + a * alpha^2 * ||g||^2
                        # a = (Delta f + alpha * ||g||^2) / (alpha^2 * ||g||^2)
                        denom = (current_alpha ** 2) * grad_norm_sq
                        if denom > 1e-12:
                            a_new = (delta_f + current_alpha * grad_norm_sq) / denom
                            if 1e-6 < a_new < 1e6:
                                a_est = a_new
                            else:
                                a_est = a_est * 0.9 # Damp update if estimate is wild
                        
                        x = x_new
                        fx = f_new
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        
                        # Update alpha for next step
                        alpha = 1.0 / (2.0 * a_est)
                        alpha = np.clip(alpha, 1e-4, 10.0)
                        break
                    else:
                        current_alpha *= 0.5
                        if current_alpha < 1e-8:
                            break
                            
                if not accepted:
                    # If we couldn't make progress, we might be stuck or at min
                    # Try a smaller random perturbation? No, stick to GD.
                    # If alpha is too small, break.
                    if alpha < 1e-6:
                        break
                    # Otherwise, keep trying with smaller alpha in next loop
                    alpha *= 0.8

        # Final Polish Phase
        # Use remaining budget to refine the best solution found
        # If we have significant budget left and are close to convergence, do a high-precision polish
        if call_count < budget - 10:
            # Check if we are in a tight region
            # Estimate gradient at best_x
            if call_count + 2 * dim <= budget:
                x_pol = best_x.copy()
                fx_pol = func(x_pol)
                call_count += 1
                
                if fx_pol < best_fx:
                    best_fx = fx_pol
                    best_x = x_pol.copy()
                
                grad = np.zeros(dim)
                for i in range(dim):
                    x_p = x_pol.copy()
                    x_m = x_pol.copy()
                    x_p[i] += 1e-8
                    x_m[i] -= 1e-8
                    f_p = func(x_p)
                    f_m = func(x_m)
                    call_count += 2
                    grad[i] = (f_p - f_m) / (2e-8)
                
                grad_norm = np.linalg.norm(grad)
                
                if grad_norm < 1e-2:
                    # Perform a few small steps with very small epsilon for gradient
                    eps_pol = 1e-8
                    alpha_pol = 1.0 / (2.0 * a_est) if a_est > 1e-6 else 0.1
                    alpha_pol = np.clip(alpha_pol, 1e-6, 1.0)
                    
                    # Simple local search with tighter tolerances
                    for _ in range(5):
                        if call_count + 2 * dim > budget:
                            break
                        
                        grad = np.zeros(dim)
                        for i in range(dim):
                            x_p = x_pol.copy()
                            x_m = x_pol.copy()
                            x_p[i] += eps_pol
                            x_m[i] -= eps_pol
                            f_p = func(x_p)
                            f_m = func(x_m)
                            call_count += 2
                            grad[i] = (f_p - f_m) / (2 * eps_pol)
                        
                        grad_norm = np.linalg.norm(grad)
                        if grad_norm < 1e-10:
                            break
                        
                        direction = -grad / grad_norm
                        # Take a small step
                        step_size = alpha_pol * grad_norm
                        x_new = x_pol + step_size * direction
                        x_new = np.clip(x_new, -5.0, 5.0)
                        
                        f_new = func(x_new)
                        call_count += 1
                        
                        if f_new < fx_pol:
                            x_pol = x_new
                            fx_pol = f_new
                            if fx_pol < best_fx:
                                best_fx = fx_pol
                                best_x = x_pol.copy()
                        else:
                            break

        return best_x, best_fx
# EVOLVE-BLOCK-END
