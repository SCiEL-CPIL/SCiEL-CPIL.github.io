"""Rewritten program using homotopy/continuation with Newton refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x of F: R^2 -> R^2 such that ||F(x)|| <= 1e-8.
    
    Uses homotopy/continuation method:
    - Track root path from x0 (trivial at t=0) to true root at t=1
    - Adaptive t-stepping with Newton refinement at each level
    - Damped Newton steps for global convergence
    - Deadline guard to stay within time limits
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
    x = np.array(x0, dtype=float)
    
    # First, evaluate F at x0 to see if we're already close
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return x
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    def compute_jacobian(x, Fx):
        """Finite-difference Jacobian with complex-step if possible, else central FD."""
        # Try complex-step first (more accurate, 4 calls)
        try:
            h = 1e-30
            J = np.zeros((2, 2))
            for j in range(2):
                x_c = x.astype(complex)
                x_c[j] += 1j * h
                Fx_c = F(x_c)
                if np.all(np.isfinite(Fx_c.real)):
                    J[:, j] = Fx_c.imag / h
                else:
                    # Fallback to central FD
                    h_fd = 1e-7
                    e = np.zeros(2)
                    e[j] = h_fd
                    J[:, j] = (F(x + e) - F(x - e)) / (2 * h_fd)
            if np.all(np.isfinite(J)):
                return J
        except (TypeError, ValueError, AttributeError):
            pass
        
        # Fallback: central finite differences
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
        return J
    
    def newton_step(x, Fx, max_iter=30):
        """Damped Newton method from current x."""
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if np.linalg.norm(Fx) <= 1e-12:
                return x, Fx
            
            J = compute_jacobian(x, Fx)
            if not np.all(np.isfinite(J)):
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Damped step with backtracking line search
            alpha = 1.0
            for _ in range(30):
                x_new = x + alpha * dx
                Fx_new = F(x_new)
                if not np.all(np.isfinite(Fx_new)):
                    alpha *= 0.5
                    continue
                if np.linalg.norm(Fx_new) < np.linalg.norm(Fx) * (1 - 1e-4 * alpha):
                    x = x_new
                    Fx = Fx_new
                    break
                alpha *= 0.5
            else:
                # If line search failed, take a small step
                x_new = x + 0.01 * dx
                Fx_new = F(x_new)
                if np.all(np.isfinite(Fx_new)):
                    x = x_new
                    Fx = Fx_new
                break
        
        return x, Fx
    
    # Homotopy: define F_t(x) = (1-t)*F(x0) + t*F(x) ... actually, 
    # better: F_t(x) = F(x) - (1-t)*F(x0) so at t=0, root is x0, at t=1, root is solution
    # Wait, that's not quite right. Let me think again.
    
    # Standard homotopy: H(x, t) = t*F(x) + (1-t)*(x - x0)
    # At t=0: H(x, 0) = x - x0, root is x0
    # At t=1: H(x, 1) = F(x), root is solution
    # This is a good choice because we start at a known root.
    
    def H(x, t):
        return t * F(x) + (1 - t) * (x - x0)
    
    # Start at t=0, x=x0
    t = 0.0
    x = np.array(x0, dtype=float)
    
    # Adaptive t-stepping
    t_steps = 0
    max_t_steps = 200
    
    while t < 1.0 and t_steps < max_t_steps:
        if time.time() > deadline:
            break
        
        # Try to increase t
        dt = 0.05  # initial step
        success = False
        
        for _ in range(20):  # max backtracking on t
            t_new = min(t + dt, 1.0)
            
            # Newton solve for H(x, t_new) = 0 starting from current x
            x_curr = x.copy()
            
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                Hx = H(x_curr, t_new)
                if np.linalg.norm(Hx) <= 1e-12:
                    break
                
                # Jacobian of H w.r.t. x
                # dH/dx = t*J_F + (1-t)*I
                # We need J_F at x_curr
                Fx_curr = F(x_curr)
                J_F = compute_jacobian(x_curr, Fx_curr)
                
                if not np.all(np.isfinite(J_F)):
                    break
                
                J_H = t_new * J_F + (1 - t_new) * np.eye(2)
                
                try:
                    dx = np.linalg.solve(J_H, -Hx)
                except np.linalg.LinAlgError:
                    break
                
                if not np.all(np.isfinite(dx)):
                    break
                
                # Damped step
                alpha = 1.0
                for _ in range(30):
                    x_new = x_curr + alpha * dx
                    Hx_new = H(x_new, t_new)
                    if not np.all(np.isfinite(Hx_new)):
                        alpha *= 0.5
                        continue
                    if np.linalg.norm(Hx_new) < np.linalg.norm(Hx) * (1 - 1e-4 * alpha):
                        x_curr = x_new
                        break
                    alpha *= 0.5
                else:
                    x_curr = x_curr + 0.01 * dx
                    break
            
            # Check if we converged at this t
            H_final = H(x_curr, t_new)
            if np.linalg.norm(H_final) <= 1e-10:
                x = x_curr
                t = t_new
                success = True
                break
            else:
                dt *= 0.5
        
        if not success:
            # Try smaller steps or break
            dt *= 0.1
            if dt < 1e-4:
                break
        
        t_steps += 1
    
    # Final Newton refinement at t=1
    if t >= 0.999 or np.linalg.norm(F(x)) <= 1e-6:
        x, Fx = newton_step(x, F(x), max_iter=50)
    
    return x
# EVOLVE-BLOCK-END
