"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # RKF45 Coefficients
    # A matrix (7x7)
    A = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.375, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 9.0/32.0, 12.0/32.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 32.0/91.0, 32.0/91.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.0]
    ])
    
    # Weights for 4th order (b)
    b = np.array([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/10.0, 0.0, 0.0])
    
    # Weights for 5th order (b_star)
    b_star = np.array([31.0/360.0, 0.0, 0.0, 512.0/1755.0, 128.0/1755.0, -1.0/30.0, 0.0])
    
    # c values
    c = np.array([0.0, 0.25, 0.375, 12.0/13.0, 1.0, 0.5, 1.0])
    
    # Tolerances
    rtol = 1e-12
    atol = 1e-14
    
    # Initial step size
    h = (t1 - t0) / 100.0
    if h == 0:
        return y
        
    # Safety factor
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    n = len(y)
    
    while t < t1:
        # Ensure we don't step past t1
        if t + h > t1:
            h = t1 - t
        
        if h <= 0:
            break
            
        # Calculate stages
        k = np.zeros((7, n))
        k[0] = f(t, y)
        
        for i in range(1, 7):
            y_stage = y.copy()
            for j in range(i):
                y_stage += h * A[i, j] * k[j]
            k[i] = f(t + c[i] * h, y_stage)
        
        # Compute 4th order solution
        y_4 = y + h * np.sum(b[:, np.newaxis] * k, axis=0)
        
        # Compute 5th order solution
        y_5 = y + h * np.sum(b_star[:, np.newaxis] * k, axis=0)
        
        # Error estimate
        err = np.linalg.norm(h * (y_5 - y_4) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_5))))
        
        if err <= 1.0:
            # Accept step
            t += h
            y = y_5 # Use higher order solution
            
            # Adjust step size
            if err > 0:
                factor = safety * (1.0 / err) ** 0.25 # 5th order method, use 1/5? 
                # For RKF45, the error is O(h^5), so we use 1/5 power?
                # Actually, standard is 1/5 for 5th order accuracy.
                # But we are using 5th order solution. The error is difference between 4 and 5.
                # The error of the 5th order solution is O(h^6).
                # So we should use 1/6?
                # Let's stick to 1/5 as a conservative estimate for the embedded pair.
                factor = np.clip(factor, min_factor, max_factor)
                h = h * factor
            else:
                h = h * max_factor
        else:
            # Reject step
            factor = safety * (1.0 / err) ** 0.25
            factor = np.clip(factor, min_factor, max_factor)
            h = h * factor
            
        # Check for overflow or non-finite
        if not np.all(np.isfinite(y)):
            return None
            
    return y
# EVOLVE-BLOCK-END
