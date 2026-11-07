"""Root finder for 3 fixed two-dimensional systems using Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Broyden's method with safeguards.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit
    
    x = np.array(x0, dtype=float).copy()
    n = 2
    
    # Initial evaluation
    Fx = np.array(F(x), dtype=float)
    norm_Fx = np.linalg.norm(Fx)
    
    if norm_Fx <= 1e-12:
        return x

    # Initialize Jacobian with finite differences
    eps = 1e-7
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        Fx_e = np.array(F(x + e), dtype=float)
        J[:, j] = (Fx_e - Fx) / eps
    
    # Broyden's method with better step control
    for i in range(100):
        if time.time() > deadline:
            break
            
        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, try a damped Newton step or break
            s = -Fx * 0.1
            if np.linalg.norm(s) > 10:
                s = s / np.linalg.norm(s) * 0.1
        
        # Step size control with backtracking line search
        alpha = 1.0
        x_new = x + alpha * s
        Fx_new = np.array(F(x_new), dtype=float)
        norm_Fx_new = np.linalg.norm(Fx_new)
        
        # Backtracking: reduce step size if norm doesn't decrease
        max_backtracks = 10
        backtrack_count = 0
        while norm_Fx_new > norm_Fx * 0.9 and backtrack_count < max_backtracks:
            alpha *= 0.5
            x_new = x + alpha * s
            Fx_new = np.array(F(x_new), dtype=float)
            norm_Fx_new = np.linalg.norm(Fx_new)
            backtrack_count += 1
        
        # If backtracking didn't help, try even smaller step or accept
        if norm_Fx_new > norm_Fx * 1.5:
            # Very bad step, try a tiny step in -Fx direction
            s_tiny = -Fx * 0.01
            x_tiny = x + s_tiny
            Fx_tiny = np.array(F(x_tiny), dtype=float)
            norm_Fx_tiny = np.linalg.norm(Fx_tiny)
            if norm_Fx_tiny < norm_Fx:
                x_new = x_tiny
                Fx_new = Fx_tiny
                norm_Fx_new = norm_Fx_tiny
                s = s_tiny
            else:
                # Accept the original step even if not great
                pass
        
        # Update Broyden
        y = Fx_new - Fx
        # Broyden update: J_new = J + (y - J s) s^T / (s^T s)
        Js = J @ s
        diff = y - Js
        s_norm_sq = np.dot(s, s)
        
        if s_norm_sq > 1e-12:
            J = J + np.outer(diff, s) / s_norm_sq
        
        x = x_new
        Fx = Fx_new
        norm_Fx = norm_Fx_new
        
        if norm_Fx <= 1e-12:
            return x
            
    return x
# EVOLVE-BLOCK-END
