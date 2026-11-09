"""Root finder for 3 fixed two-dimensional systems using Levenberg-Marquardt with complex-step Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """Find x in R^2 such that ||F(x)|| <= 1e-8 using LM with complex-step Jacobian."""
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill
    
    x = np.array(x0, dtype=float).copy()
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    # First evaluation
    try:
        Fx = np.asarray(F(x), dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            # Try a few perturbations
            for trial in range(10):
                x_try = x + np.random.randn(2) * 0.1 * (trial + 1)
                try:
                    Fx_try = np.asarray(F(x_try), dtype=float)
                    if Fx_try.shape == (2,) and np.all(np.isfinite(Fx_try)):
                        x = x_try
                        Fx = Fx_try
                        break
                except Exception:
                    continue
            else:
                return x
    except Exception:
        # Try a few perturbations
        for trial in range(10):
            x_try = x + np.random.randn(2) * 0.1 * (trial + 1)
            try:
                Fx_try = np.asarray(F(x_try), dtype=float)
                if Fx_try.shape == (2,) and np.all(np.isfinite(Fx_try)):
                    x = x_try
                    Fx = Fx_try
                    break
            except Exception:
                continue
        else:
            return x
    
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    # Levenberg-Marquardt parameters
    lam = 1e-3  # damping factor
    lam_min = 1e-12
    lam_max = 1e12
    max_iter = 200
    
    for iteration in range(max_iter):
        # Check deadline
        if time.time() > DEADLINE:
            break
        
        # Compute Jacobian using complex-step differentiation (2 calls)
        J = np.zeros((2, 2))
        h = 1e-30  # complex step size
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            # F(x + h*e_j) where x is real, but we add complex
            # Since F may not support complex, use central difference with small h
            # Actually, let's use a small real step for compatibility
            h_real = 1e-8
            e_real = np.zeros(2)
            e_real[j] = h_real
            try:
                F_plus = np.asarray(F(x + e_real), dtype=float)
                F_minus = np.asarray(F(x - e_real), dtype=float)
                if F_plus.shape == (2,) and np.all(np.isfinite(F_plus)) and \
                   F_minus.shape == (2,) and np.all(np.isfinite(F_minus)):
                    J[:, j] = (F_plus - F_minus) / (2 * h_real)
                else:
                    # Fallback to forward difference
                    F_plus = np.asarray(F(x + e_real), dtype=float)
                    if F_plus.shape == (2,) and np.all(np.isfinite(F_plus)):
                        J[:, j] = (F_plus - Fx) / h_real
                    else:
                        J[:, j] = 0.0
            except Exception:
                J[:, j] = 0.0
        
        # Check if Jacobian is valid
        if not np.all(np.isfinite(J)):
            # Try steepest descent step
            grad = J.T @ Fx  # gradient of 0.5*||F||^2
            if np.linalg.norm(grad) > 0:
                step = -grad / (np.linalg.norm(grad) ** 2)
            else:
                step = -Fx / (np.linalg.norm(Fx) + 1e-15)
            x_new = x + 0.1 * step
            try:
                F_new = np.asarray(F(x_new), dtype=float)
                if F_new.shape == (2,) and np.all(np.isfinite(F_new)):
                    norm_new = np.linalg.norm(F_new)
                    if norm_new < best_norm:
                        x = x_new
                        Fx = F_new
                        best_x = x.copy()
                        best_norm = norm_new
                    if norm_new <= 1e-8:
                        return x
            except Exception:
                pass
            continue
        
        # Solve (J^T J + lam * I) dx = -J^T F
        JtJ = J.T @ J
        I = np.eye(2)
        A = JtJ + lam * I
        b = -J.T @ Fx
        
        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            try:
                dx = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception:
                dx = -J.T @ Fx / (np.linalg.norm(J.T @ Fx) + 1e-15)
        
        if not np.all(np.isfinite(dx)):
            dx = -J.T @ Fx / (np.linalg.norm(J.T @ Fx) + 1e-15)
        
        # Try the step with line search
        alpha = 1.0
        x_new = x + alpha * dx
        accepted = False
        
        for _ in range(20):
            try:
                F_new = np.asarray(F(x_new), dtype=float)
                if F_new.shape == (2,) and np.all(np.isfinite(F_new)):
                    norm_new = np.linalg.norm(F_new)
                    if norm_new <= 1e-8:
                        x = x_new
                        Fx = F_new
                        return x
                    if norm_new < best_norm:
                        best_x = x_new.copy()
                        best_norm = norm_new
                    # Accept if we made progress
                    if norm_new < np.linalg.norm(Fx):
                        x = x_new
                        Fx = F_new
                        accepted = True
                        break
            except Exception:
                pass
            
            alpha *= 0.5
            x_new = x + alpha * dx
            if alpha < 1e-15:
                break
        
        if accepted:
            # Decrease lambda (step was good)
            lam = max(lam / 10.0, lam_min)
        else:
            # Increase lambda (step was bad)
            lam = min(lam * 10.0, lam_max)
        
        # If lambda is very large, we're essentially doing gradient descent
        if lam > 1e10:
            # Try a small gradient step
            grad = J.T @ Fx
            if np.linalg.norm(grad) > 1e-15:
                step = -grad * 0.01
                x_new = x + step
                try:
                    F_new = np.asarray(F(x_new), dtype=float)
                    if F_new.shape == (2,) and np.all(np.isfinite(F_new)):
                        norm_new = np.linalg.norm(F_new)
                        if norm_new < best_norm:
                            x = x_new
                            Fx = F_new
                            best_x = x.copy()
                            best_norm = norm_new
                        if norm_new <= 1e-8:
                            return x
                except Exception:
                    pass
    
    # Final check
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Return the best point found
    if best_norm < np.linalg.norm(Fx):
        return best_x
    
    return x
# EVOLVE-BLOCK-END
