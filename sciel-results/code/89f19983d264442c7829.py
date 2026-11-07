"""Root finder for 3 fixed 2D systems using Homotopy (Tracing) with Broyden fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Primary strategy: Trace the zero-level set of F[0] to find intersection with F[1]=0.
    Fallback strategy: Broyden's method.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    max_total_calls = 450
    
    call_count = 0
    
    def F_wrapper(x):
        nonlocal call_count
        if call_count >= max_total_calls:
            return np.array([0.0, 0.0])
        call_count += 1
        return np.array(F(x))
    
    def time_up():
        return time.time() > deadline

    # --- Phase 1: Homotopy / Contour Tracing ---
    
    def find_start_on_F1():
        """Find a point where F1(x) is close to 0."""
        # Try different radii around x0
        best_x = np.array(x0, dtype=float)
        best_val = np.inf
        
        # Sample points on circles
        for radius in [0.1, 1.0, 10.0, 100.0]:
            for theta in np.linspace(0, 2*np.pi, 20, endpoint=False):
                x_test = np.array([
                    x0[0] + radius * np.cos(theta),
                    x0[1] + radius * np.sin(theta)
                ])
                if not np.all(np.isfinite(x_test)):
                    continue
                Fx = F_wrapper(x_test)
                if not np.all(np.isfinite(Fx)):
                    continue
                val = abs(Fx[0])
                if val < best_val:
                    best_val = val
                    best_x = x_test
        
        # Refine best_x to satisfy F1(x)=0 using 1D Newton along normal
        # Get gradient of F1 at best_x
        h = 1e-6
        Fx0 = F_wrapper(best_x)
        Fx1 = F_wrapper(best_x + np.array([h, 0]))
        Fx2 = F_wrapper(best_x + np.array([0, h]))
        grad_F1 = np.array([Fx1[0] - Fx0[0], Fx2[0] - Fx0[0]]) / h
        
        norm_grad = np.linalg.norm(grad_F1)
        if norm_grad < 1e-10:
            return best_x, 0 # Cannot refine
        
        normal = grad_F1 / norm_grad
        
        # 1D Newton on g(beta) = F1(best_x + beta * normal)
        beta = 0.0
        for _ in range(10):
            x_curr = best_x + beta * normal
            Fx = F_wrapper(x_curr)
            if not np.all(np.isfinite(Fx)):
                break
            g_val = Fx[0]
            if abs(g_val) < 1e-12:
                break
            
            # Derivative of F1 along normal
            # d/dbeta F1(x + beta*n) = n^T * grad_F1(x + beta*n)
            # Approximate grad at x_curr
            Fx_p1 = F_wrapper(x_curr + np.array([h, 0]))
            Fx_p2 = F_wrapper(x_curr + np.array([0, h]))
            grad_curr = np.array([Fx_p1[0] - Fx[0], Fx_p2[0] - Fx[0]]) / h
            dg_dbeta = np.dot(normal, grad_curr)
            
            if abs(dg_dbeta) < 1e-10:
                break
            
            beta_new = beta - g_val / dg_dbeta
            if abs(beta_new - beta) < 1e-10:
                beta = beta_new
                break
            beta = beta_new
            
        return best_x + beta * normal, 1 # 1 indicates refinement attempted

    def trace_F1(x_start, direction_sign, max_steps=40):
        """
        Trace F1(x)=0 starting from x_start.
        direction_sign: +1 or -1 for tangent direction.
        """
        x = np.array(x_start, dtype=float)
        
        for i in range(max_steps):
            if time_up() or call_count >= max_total_calls:
                return None
            
            # Check if we found a root
            Fx = F_wrapper(x)
            if not np.all(np.isfinite(Fx)):
                return None
            if np.linalg.norm(Fx) <= 1e-8:
                return x
            
            # Compute gradient of F1
            h = 1e-6
            Fx0 = F_wrapper(x)
            Fx1 = F_wrapper(x + np.array([h, 0]))
            Fx2 = F_wrapper(x + np.array([0, h]))
            grad_F1 = np.array([Fx1[0] - Fx0[0], Fx2[0] - Fx0[0]]) / h
            
            norm_grad = np.linalg.norm(grad_F1)
            if norm_grad < 1e-10:
                # Singular point, try to escape or break
                return None
                
            normal = grad_F1 / norm_grad
            # Tangent is perpendicular to normal
            tangent = np.array([-normal[1], normal[0]])
            
            # Choose step size
            # Adaptive step: start with 0.1, reduce if correction is large
            step_size = 0.1 
            
            # Predict
            x_pred = x + direction_sign * step_size * tangent
            
            # Correct to stay on F1(x)=0
            # Solve F1(x_pred + beta * normal) = 0
            beta = 0.0
            for _ in range(5):
                x_curr = x_pred + beta * normal
                Fx_curr = F_wrapper(x_curr)
                if not np.all(np.isfinite(Fx_curr)):
                    return None
                g_val = Fx_curr[0]
                if abs(g_val) < 1e-12:
                    break
                
                # Derivative
                Fx_p1 = F_wrapper(x_curr + np.array([h, 0]))
                Fx_p2 = F_wrapper(x_curr + np.array([0, h]))
                grad_curr = np.array([Fx_p1[0] - Fx_curr[0], Fx_p2[0] - Fx_curr[0]]) / h
                dg_dbeta = np.dot(normal, grad_curr)
                
                if abs(dg_dbeta) < 1e-10:
                    break
                
                beta_new = beta - g_val / dg_dbeta
                if abs(beta_new - beta) < 1e-8:
                    beta = beta_new
                    break
                beta = beta_new
                
            x_new = x_pred + beta * normal
            
            # Check if we made progress
            if not np.all(np.isfinite(x_new)):
                return None
                
            # Check if F2 is small
            Fx_new = F_wrapper(x_new)
            if not np.all(np.isfinite(Fx_new)):
                return None
            if np.linalg.norm(Fx_new) <= 1e-8:
                return x_new
                
            # Update x
            x = x_new
            
        return None

    # 1. Find start point
    x_start, _ = find_start_on_F1()
    
    # 2. Trace in + direction
    x_root = trace_F1(x_start, 1)
    if x_root is not None:
        return x_root
        
    # 3. Trace in - direction
    x_root = trace_F1(x_start, -1)
    if x_root is not None:
        return x_root

    # --- Phase 2: Fallback Broyden's Method ---
    
    def broyden(F, x0, max_iter=50):
        x = np.array(x0, dtype=float)
        # Initial Jacobian estimate (finite diff)
        h = 1e-6
        Fx0 = F_wrapper(x)
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F_wrapper(x + e) - Fx0) / h
            
        for _ in range(max_iter):
            if time_up() or call_count >= max_total_calls:
                break
                
            Fx = F_wrapper(x)
            if not np.all(np.isfinite(Fx)):
                break
            if np.linalg.norm(Fx) <= 1e-8:
                return x
                
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
                
            # Line search
            alpha = 1.0
            Fx_norm = np.linalg.norm(Fx)
            for _ in range(10):
                x_trial = x + alpha * s
                Fx_trial = F_wrapper(x_trial)
                if not np.all(np.isfinite(Fx_trial)):
                    alpha *= 0.5
                    continue
                Fx_trial_norm = np.linalg.norm(Fx_trial)
                if Fx_trial_norm < Fx_norm * (1 - 1e-4 * alpha): # Armijo
                    break
                alpha *= 0.5
                
            x_new = x + alpha * s
            Fx_new = F_wrapper(x_new)
            
            if not np.all(np.isfinite(Fx_new)):
                break
                
            # Broyden Update
            y = Fx_new - Fx
            s_vec = x_new - x
            s_norm_sq = np.dot(s_vec, s_vec)
            if s_norm_sq > 1e-12:
                J = J + np.outer(y - J @ s_vec, s_vec) / s_norm_sq
                
            x = x_new
            
        return x

    x_root = broyden(F_wrapper, x0)
    if x_root is not None:
        return x_root

    return np.array(x0)
# EVOLVE-BLOCK-END
