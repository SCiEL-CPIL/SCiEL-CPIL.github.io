"""Root finder for 3 fixed two-dimensional systems using Gauss-Newton with robust safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _safe_F(F, x):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(x)
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using Gauss-Newton method
    minimizing g(x) = ||F(x)||^2.
    
    Uses finite-difference Jacobian with adaptive step control and
    robust safeguards including deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.array([0.0, 0.0])
    
    # Try to get initial F evaluation
    Fx = _safe_F(F, x)
    if Fx is None:
        # Try a few perturbed points
        for dx in [1e-3, -1e-3, 1e-2, -1e-2, 1.0, -1.0]:
            x_try = x + dx
            Fx = _safe_F(F, x_try)
            if Fx is not None:
                x = x_try
                break
        if Fx is None:
            return x
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    # Check if already at root
    if best_norm <= 1e-8:
        return x
    
    # Gauss-Newton: minimize g(x) = ||F(x)||^2
    # Gradient of g: J^T * F
    # Hessian approx: J^T * J
    
    for iteration in range(100):
        # Deadline check
        if time.time() > deadline:
            break
        
        # Compute finite-difference Jacobian
        eps = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            Fx_plus = _safe_F(F, x + e)
            if Fx_plus is None:
                # Try smaller step
                eps = 1e-8
                e[j] = eps
                Fx_plus = _safe_F(F, x + e)
                if Fx_plus is None:
                    # Try backward difference
                    e[j] = -eps
                    Fx_plus = _safe_F(F, x + e)
                    if Fx_plus is None:
                        break
                    J[:, j] = (Fx - Fx_plus) / eps
                else:
                    J[:, j] = (Fx_plus - Fx) / eps
            else:
                J[:, j] = (Fx_plus - Fx) / eps
        else:
            # Successfully computed all columns
            pass
        if time.time() > deadline:
            break
        
        # Check if Jacobian is well-conditioned
        try:
            # Solve J^T J delta = -J^T F
            JTJ = J.T @ J
            JTF = J.T @ Fx
            
            # Check condition number
            eigvals = np.linalg.eigvalsh(JTJ)
            if np.min(eigvals) < 1e-15:
                # Add regularization (Levenberg-Marquardt style)
                lambda_reg = 1e-6
                JTJ += lambda_reg * np.eye(2)
            
            delta = np.linalg.solve(JTJ, -JTF)
        except np.linalg.LinAlgError:
            # Fall back to steepest descent
            grad = J.T @ Fx
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-15:
                break
            delta = -grad / norm_grad
            # Scale step
            step_size = min(1.0, 1.0 / norm_grad)
            delta *= step_size
        
        # Line search with backtracking
        alpha = 1.0
        Fx_norm_sq = np.dot(Fx, Fx)
        grad = J.T @ Fx
        directional_deriv = np.dot(grad, delta)
        
        max_backtracks = 20
        for _bt in range(max_backtracks):
            if time.time() > deadline:
                break
            
            x_new = x + alpha * delta
            Fx_new = _safe_F(F, x_new)
            if Fx_new is None:
                alpha *= 0.5
                continue
            
            Fx_new_norm_sq = np.dot(Fx_new, Fx_new)
            
            # Armijo condition
            if Fx_new_norm_sq <= Fx_norm_sq + 1e-4 * alpha * directional_deriv:
                # Accept step
                x = x_new
                Fx = Fx_new
                Fx_norm = np.linalg.norm(Fx)
                
                if Fx_norm < best_norm:
                    best_x = x.copy()
                    best_norm = Fx_norm
                
                if Fx_norm <= 1e-8:
                    return x
                
                # Update Fx_norm_sq for next iteration
                Fx_norm_sq = Fx_new_norm_sq
                break
            else:
                alpha *= 0.5
        
        # If no step was accepted, try smaller perturbation
        if time.time() > deadline:
            break
        
        # Check progress
        if np.linalg.norm(delta) < 1e-12:
            break
    
    # Return the best point found
    if best_norm < np.linalg.norm(_safe_F(F, x) if _safe_F(F, x) is not None else np.array([0,0])):
        return best_x
    return x
# EVOLVE-BLOCK-END
