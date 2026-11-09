"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Adaptive RK4(5) solver for y' = f(t, y).
    Uses the Bogacki-Shampine 4(5) pair for efficiency and stability.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Safety deadline: 290 seconds (leaves margin for evaluator overhead)
    deadline = time.time() + 290.0
    
    # Initial step size estimate
    # Heuristic: assume O(1) scale, try h=1e-3 initially
    h = 1e-3 * (t1 - t0)
    if h <= 0:
        h = 1e-6
    # Cap initial step to not overshoot
    h = min(h, t1 - t0)
    
    # Tolerance settings
    # We want high accuracy. Relative error ~1e-12 to 1e-13.
    # Score is -log10(err) - 0.5 log10(calls).
    # If err ~ 1e-13, -log10(err) ~ 13.
    # If calls ~ 1000, -0.5 log10(1000) ~ -1.5. Score ~ 11.5.
    # If we use RK4 fixed 2000 steps:
    #   Calls = 8000. -0.5 log10(8000) ~ -1.9.
    #   Err for RK4 with h=2.5e-5 on heat eq? 
    #   Heat eq is stiff? 10x10 grid. dx=1/9? 
    #   Usually stable for explicit RK if h < dx^2 / (2 * d) where d is dimension.
    #   Let's rely on adaptive step control to find the optimal balance.
    
    rtol = 1e-12
    atol = 1e-14
    
    # Bogacki-Shampine 4(5) coefficients
    # k1 = f(t, y)
    # k2 = f(t + h/2, y + h/2 * k1)
    # k3 = f(t + 3h/4, y + 9h/32 * k1 + 9h/32 * k2)
    # k4 = f(t + h, y + 1932h/2197 * k1 - 7200h/2197 * k2 + 7296h/2197 * k3)
    # k5 = f(t + h/2, y + 439h/4104 * k1 - 1800h/827 * k2 + 7296h/827 * k3 - 277h/5103 * k4)
    # y_{n+1} = y + h * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5)  [4th order]
    # y_{n+1}_5 = y + h * (5179/57600 * k1 - 7571/16695 * k2 + 29583/33920 * k3 - 9/40 * k4 + 181/720 * k5) [5th order]
    # Error = y_5 - y_4
    
    # Coeffs for k2
    c2 = 0.5
    a21 = 0.5
    # Coeffs for k3
    c3 = 0.75
    a31 = 9.0 / 32.0
    a32 = 9.0 / 32.0
    # Coeffs for k4
    c4 = 1.0
    a41 = 1932.0 / 2197.0
    a42 = -7200.0 / 2197.0
    a43 = 7296.0 / 2197.0
    # Coeffs for k5
    c5 = 0.5
    a51 = 439.0 / 4104.0
    a52 = -1800.0 / 827.0
    a53 = 7296.0 / 827.0
    a54 = -277.0 / 5103.0
    
    # Coeffs for 4th order update
    b4_1 = 35.0 / 384.0
    b4_2 = 0.0
    b4_3 = 500.0 / 1113.0
    b4_4 = 125.0 / 192.0
    b4_5 = -2187.0 / 6784.0
    
    # Coeffs for 5th order update (for error estimation, we usually just compute diff)
    # Actually, for embedded method, we compute y_new using 4th order coeffs and estimate error using 5th order coeffs.
    # Or compute both and take 5th order if accepted, else 4th? Standard is to use 4th order for step, 5th for error check.
    # y_{n+1} = y_n + h * sum(b4_i * k_i)
    # err = h * sum((b5_i - b4_i) * k_i)
    
    b5_1 = 5179.0 / 57600.0
    b5_2 = -7571.0 / 16695.0
    b5_3 = 29583.0 / 33920.0
    b5_4 = -9.0 / 40.0
    b5_5 = 181.0 / 720.0
    
    # Difference b5 - b4
    db_1 = b5_1 - b4_1
    db_2 = b5_2 - b4_2
    db_3 = b5_3 - b4_3
    db_4 = b5_4 - b4_4
    db_5 = b5_5 - b4_5
    
    n = len(y)
    
    while t < t1:
        if time.time() > deadline:
            # Return current best estimate if time runs out
            return y
            
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
            
        if h <= 0:
            break
            
        k1 = f(t, y)
        k2 = f(t + c2 * h, y + h * a21 * k1)
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        
        # 4th order update
        y_new = y + h * (b4_1 * k1 + b4_3 * k3 + b4_4 * k4 + b4_5 * k5)
        
        # Error estimate (norm of difference between 5th and 4th order)
        # err_vec = h * (db_1 * k1 + db_2 * k2 + db_3 * k3 + db_4 * k4 + db_5 * k5)
        err_vec = h * (db_1 * k1 + db_2 * k2 + db_3 * k3 + db_4 * k4 + db_5 * k5)
        
        # Calculate norm
        # Use RMS norm for stability
        scale = np.sqrt(np.mean(np.abs(y)**2)) if np.any(y != 0) else 1.0
        # Simple Euclidean norm is often fine, but scaled RMS is robust
        err_norm = np.sqrt(np.mean(err_vec**2))
        
        # Tolerance
        tol = rtol * scale + atol
        
        if err_norm <= tol:
            # Accept step
            t += h
            y = y_new
            
            # Adjust step size
            if err_norm > 0:
                # Safety factor 0.9
                factor = 0.9 * (tol / err_norm) ** 0.2
            else:
                factor = 5.0 # Max growth if error is zero
            
            # Limit growth
            if factor > 5.0:
                factor = 5.0
            elif factor < 0.1:
                factor = 0.1
                
            h_new = h * factor
        else:
            # Reject step
            if err_norm > 0:
                factor = 0.9 * (tol / err_norm) ** 0.25
            else:
                factor = 0.1
                
            if factor < 0.1:
                factor = 0.1
            elif factor > 0.9:
                factor = 0.9
                
            h_new = h * factor
            
        h = h_new
        
    return y
# EVOLVE-BLOCK-END
