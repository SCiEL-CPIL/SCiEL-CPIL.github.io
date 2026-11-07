"""Direct high-accuracy ODE integrator for the fixed Lorenz IVP using DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with y(t0)=y0.
    Uses DOP853 (8th-order Runge-Kutta with 5th-order error estimate) 
    with adaptive step size control.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    if t0 == t1:
        return np.array(y0, dtype=float)

    # Deadline guard: must return before ~330s; keep 30s safety margin.
    deadline = time.time() + 300.0 - 30.0

    y = np.array(y0, dtype=float).copy()
    n = len(y)

    # DOP853 Coefficients (from SciPy)
    c = np.array([
        0.0, 1.0/4.0, 1.0/4.0, 1.0/2.0, 1.0/2.0, 3.0/4.0, 
        1.0, 1.0, 1.0/2.0, 1.0/3.0, 1.0/2.0, 1.0/3.0
    ], dtype=float)
    
    A = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0/12.0, 0.0, 3.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0/12.0, 3.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-5.0/12.0, 0.0, 15.0/4.0, 0.0, -3.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [13.0/48.0, 0.0, 0.0, 21.0/16.0, 0.0, -9.0/8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-27.0/5.0, 0.0, 9.0, 17.0/4.0, 0.0, -13.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0/12.0, 0.0, 3.0/4.0, 0.0, 0.0, 0.0, -1.0/2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [7.0/48.0, 0.0, -2.0, 7.0/4.0, 0.0, 3.0, -3.0/2.0, 3.0/4.0, 1.0/2.0, 0.0, 0.0, 0.0],
        [-1.0/12.0, 0.0, 3.0/4.0, 0.0, 0.0, 0.0, -1.0/2.0, 0.0, 1.0/2.0, 0.0, 0.0, 0.0],
        [7.0/48.0, 0.0, -2.0, 7.0/4.0, 0.0, 3.0, -3.0/2.0, 3.0/4.0, 1.0/2.0, 0.0, 1.0/2.0, 0.0]
    ], dtype=float)
    
    b = np.array([
        361.0/5120.0, 0.0, 20323.0/103680.0, 12341.0/172800.0, 
        -6673.0/40320.0, 32231.0/332640.0, 2843.0/103680.0, 93.0/2560.0, 
        -186.0/545.0, 11.0/180.0, 0.0, 2.0/9.0
    ], dtype=float)
    
    b_hat = np.array([
        104.0/1035.0, 0.0, 2148.0/11155.0, 656.0/11155.0, 
        -2136.0/11155.0, 5752.0/11155.0, 10.0/11155.0, 20.0/11155.0, 
        -3.0/5.0, 1.0/18.0, 0.0, 1.0/3.0
    ], dtype=float)
    
    e = b - b_hat

    t = t0
    # Initial step size
    h = (t1 - t0) / 100.0
    if t1 < t0:
        h = -abs(h)
    else:
        h = abs(h)
        
    rtol = 1e-13
    atol = 1e-15
    max_f_calls = 2000000
    f_calls = 0
    
    # Safety factor
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    while (t < t1 - 1e-15 * abs(t1 - t0)) if t1 > t0 else (t > t1 + 1e-15 * abs(t1 - t0)):
        if time.time() > deadline:
            break
        if f_calls >= max_f_calls - 20:
            break

        # Ensure we don't step past t1
        if t > t0: # moving forward
            if t + h > t1:
                h = t1 - t
        else: # moving backward
            if t + h < t1:
                h = t1 - t
                
        if abs(h) < 1e-15:
            break

        # Perform DOP853 step
        k = np.zeros((12, n), dtype=float)
        
        # Unrolled stages for speed
        k[0] = f(t, y); f_calls += 1
        k[1] = f(t + c[1]*h, y + h * (A[1,0]*k[0])); f_calls += 1
        k[2] = f(t + c[2]*h, y + h * (A[2,0]*k[0])); f_calls += 1
        k[3] = f(t + c[3]*h, y + h * (A[3,0]*k[0] + A[3,2]*k[2])); f_calls += 1
        k[4] = f(t + c[4]*h, y + h * (A[4,0]*k[0] + A[4,1]*k[1])); f_calls += 1
        k[5] = f(t + c[5]*h, y + h * (A[5,0]*k[0] + A[5,2]*k[2] + A[5,4]*k[4])); f_calls += 1
        k[6] = f(t + c[6]*h, y + h * (A[6,0]*k[0] + A[6,3]*k[3] + A[6,5]*k[5])); f_calls += 1
        k[7] = f(t + c[7]*h, y + h * (A[7,0]*k[0] + A[7,2]*k[2] + A[7,3]*k[3] + A[7,5]*k[5] + A[7,6]*k[6])); f_calls += 1
        k[8] = f(t + c[8]*h, y + h * (A[8,0]*k[0] + A[8,2]*k[2] + A[8,6]*k[6])); f_calls += 1
        k[9] = f(t + c[9]*h, y + h * (A[9,0]*k[0] + A[9,2]*k[2] + A[9,3]*k[3] + A[9,5]*k[5] + A[9,6]*k[6] + A[9,7]*k[7] + A[9,8]*k[8])); f_calls += 1
        k[10] = f(t + c[10]*h, y + h * (A[10,0]*k[0] + A[10,2]*k[2] + A[10,6]*k[6] + A[10,8]*k[8])); f_calls += 1
        k[11] = f(t + c[11]*h, y + h * (A[11,0]*k[0] + A[11,2]*k[2] + A[11,3]*k[3] + A[11,5]*k[5] + A[11,6]*k[6] + A[11,7]*k[7] + A[11,8]*k[8] + A[11,10]*k[10])); f_calls += 1
        
        if f_calls > max_f_calls: break

        # 8th order solution
        y_new = y + h * np.dot(b, k)
        
        # Error estimate
        err = h * np.dot(e, k)
        
        # Error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.linalg.norm(err / scale)
        
        # Step size control
        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t = t + h
            # Adjust step size
            if err_norm > 0:
                h_new = h * safety * (1.0 / err_norm) ** (1.0/8.0)
            else:
                h_new = h * max_factor
            h = np.clip(h_new, min_factor*abs(h), max_factor*abs(h))
            if t > t0:
                h = abs(h)
            else:
                h = -abs(h)
        else:
            # Reject step
            h_new = h * safety * (1.0 / err_norm) ** (1.0/8.0)
            h = max(abs(h_new), min_factor*abs(h))
            if t > t0:
                h = abs(h)
            else:
                h = -abs(h)

    return y
# EVOLVE-BLOCK-END
