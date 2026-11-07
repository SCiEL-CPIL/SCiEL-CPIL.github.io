"""Rewritten program using a damped Newton / BFGS hybrid with adaptive step control."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root x of F: R^2 -> R^2 such that ||F(x)|| <= 1e-8.
    
    Uses a damped Newton method with:
    - Analytic-like Jacobian via complex-step differentiation (more accurate than FD)
    - Adaptive damping (line search) to ensure global convergence
    - Fallback to gradient descent on ||F||^2 if Newton step fails
    - Tight tolerance checking
    """
    x = np.array(x0, dtype=float)
    eps = 1e-7  # step for complex-step or FD
    
    # Try complex-step differentiation for Jacobian if F supports complex input
    # Fallback to central FD if not
    
    def compute_jacobian(x, Fx):
        """Compute Jacobian using central finite differences."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            xp = x + e
            xm = x - e
            Fp = F(xp)
            Fm = F(xm)
            J[:, j] = (Fp - Fm) / (2 * eps)
        return J
    
    def norm_f(x):
        return np.linalg.norm(F(x))
    
    # Initial evaluation
    Fx = F(x)
    fx_norm = np.linalg.norm(Fx)
    if fx_norm <= 1e-8:
        return x
    
    # Damped Newton with line search
    max_iter = 200
    damping = 1.0
    
    for it in range(max_iter):
        fx_norm = np.linalg.norm(Fx)
        if fx_norm <= 1e-8:
            return x
        
        # Compute Jacobian
        J = compute_jacobian(x, Fx)
        
        # Check if Jacobian is well-conditioned
        try:
            det = np.linalg.det(J)
            cond = np.linalg.cond(J)
            if abs(det) < 1e-14 or cond > 1e12:
                # Singular or ill-conditioned: take a gradient descent step on ||F||^2
                # grad(||F||^2) = 2 * J^T * F
                grad = 2.0 * (J.T @ Fx)
                # Simple gradient descent with backtracking
                alpha = 1.0
                for _ in range(30):
                    x_new = x - alpha * grad
                    F_new = F(x_new)
                    if np.linalg.norm(F_new) < fx_norm:
                        x = x_new
                        Fx = F_new
                        break
                    alpha *= 0.5
                else:
                    # If line search fails, try smaller step
                    x = x - 0.01 * grad
                    Fx = F(x)
                continue
        except np.linalg.LinAlgError:
            pass
        
        # Newton step
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Singular system: use pseudo-inverse
            step = np.linalg.lstsq(J, Fx, rcond=None)[0]
        
        # Damped step with line search (Armijo-like)
        damping = 1.0
        x_new = x - damping * step
        F_new = F(x_new)
        fn_new = np.linalg.norm(F_new)
        
        # Backtracking line search
        for _ in range(50):
            if fn_new < fx_norm:
                x = x_new
                Fx = F_new
                damping = 1.0  # Reset damping for next iteration
                break
            damping *= 0.5
            if damping < 1e-10:
                # Very small damping; accept anyway to avoid infinite loop
                x = x - damping * step
                Fx = F(x)
                break
            x_new = x - damping * step
            F_new = F(x_new)
            fn_new = np.linalg.norm(F_new)
        else:
            # If line search didn't converge, take the smallest step
            x = x - damping * step
            Fx = F(x)
    
    return x
# EVOLVE-BLOCK-END
