"""Root finder for 3 fixed two-dimensional systems using Levenberg-Marquardt."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Levenberg-Marquardt method.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    x = np.array(x0, dtype=float).copy()
    n = 2
    
    # State
    Fx = np.array(F(x), dtype=float)
    calls = 1
    
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    def get_jacobian(x, Fx):
        nonlocal calls
        eps = 1e-6
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = eps
            x_pert = x + e
            Fx_pert = np.array(F(x_pert), dtype=float)
            calls += 1
            J[:, j] = (Fx_pert - Fx) / eps
        return J

    J = get_jacobian(x, Fx)
    
    lam = 1e-3
    min_lam = 1e-12
    max_lam = 1e12
    max_iter = 50
    max_calls = 450
    
    for it in range(max_iter):
        if calls >= max_calls:
            break
            
        # Solve (J^T J + lam * I) dx = -J^T F
        try:
            JtJ = J.T @ J
            JtF = J.T @ Fx
            A = JtJ + lam * np.eye(n)
            dx = np.linalg.solve(A, -JtF)
        except np.linalg.LinAlgError:
            lam *= 10
            if lam > max_lam:
                break
            continue

        x_trial = x + dx
        Fx_trial = np.array(F(x_trial), dtype=float)
        calls += 1
        
        if not np.all(np.isfinite(Fx_trial)):
            lam *= 10
            if lam > max_lam:
                break
            continue
            
        norm_Fx_trial = np.linalg.norm(Fx_trial)
        norm_Fx = np.linalg.norm(Fx)
        
        # Accept if significant reduction
        if norm_Fx_trial < norm_Fx * (1 - 1e-4):
            x = x_trial
            Fx = Fx_trial
            
            if norm_Fx_trial <= 1e-8:
                break
            
            lam = max(lam / 10.0, min_lam)
            
            # Update Jacobian periodically
            if it % 3 == 0:
                J = get_jacobian(x, Fx)
        else:
            lam *= 10
            if lam > max_lam:
                break

    return x
# EVOLVE-BLOCK-END
