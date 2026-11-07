"""Root finder for 3 fixed two-dimensional systems using Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Broyden's method.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
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
    
    # Broyden's method
    for i in range(100):
        if time.time() > deadline:
            break
            
        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, try a damped Newton step or break
            # For 2x2, we can try pseudo-inverse or just break
            # Let's try a small step in the direction of -Fx if J is bad
            s = -Fx * 0.1
            if np.linalg.norm(s) > 10:
                s = s / np.linalg.norm(s) * 0.1
        
        # Step size control (optional, but Broyden is usually good)
        # We can use a simple line search or just accept the step
        # For robustness, let's try a simple backtracking if norm increases
        x_new = x + s
        Fx_new = np.array(F(x_new), dtype=float)
        norm_Fx_new = np.linalg.norm(Fx_new)
        
        # If the new point is much worse, try a smaller step
        # Simple damping: if norm increases, halve the step
        if norm_Fx_new > norm_Fx * 1.1:
            # Try half step
            s_half = s * 0.5
            x_half = x + s_half
            Fx_half = np.array(F(x_half), dtype=float)
            norm_Fx_half = np.linalg.norm(Fx_half)
            if norm_Fx_half < norm_Fx_new:
                x_new = x_half
                Fx_new = Fx_half
                norm_Fx_new = norm_Fx_half
                s = s_half
            # If still worse, we might just accept or try another damping
            # For simplicity, we accept the best of the two
        
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
