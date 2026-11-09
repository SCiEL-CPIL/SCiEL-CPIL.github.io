import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Dormand-Prince coefficients (RK45)
    a2 = 1/5
    a3 = 3/10
    a4 = 4/5
    a5 = 8/9
    a6 = 1.0
    a7 = 1.0
    
    b21 = 1/5
    b31 = 3/40
    b32 = 9/40
    b41 = 44/45
    b42 = -56/15
    b43 = 32/9
    b51 = 19372/6561
    b52 = -25360/2187
    b53 = 64448/6561
    b54 = -212/729
    b61 = 9017/3168
    b62 = -355/33
    b63 = 46732/5247
    b64 = 49/176
    b65 = -5103/18656
    
    b71 = 35/384
    b73 = 500/1113
    b74 = 125/192
    b76 = -2187/6784
    
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    c7 = 1.0
    
    # Error coefficients
    e1 = 71/57600
    e3 = -71/16695
    e4 = 71/1920
    e5 = -17253/339200
    e6 = 22/525
    e7 = -1/40
    
    rtol = 1e-10
    atol = 1e-12
    
    # Initial step size
    h = (t1 - t0) / 100.0
    t = t0
    calls = 0
    start_time = time.time()
    
    while t < t1:
        # Don't step past t1
        if t + h > t1:
            h = t1 - t
        if h <= 0:
            break
            
        # Evaluate stages
        k1 = f(t, y)
        calls += 1
        
        k2 = f(t + c2*h, y + h*b21*k1)
        calls += 1
        
        k3 = f(t + c3*h, y + h*(b31*k1 + b32*k2))
        calls += 1
        
        k4 = f(t + c4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        calls += 1
        
        k5 = f(t + c5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        calls += 1
        
        k6 = f(t + c6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        calls += 1
        
        # 4th order solution
        y_new4 = y + h*(b71*k1 + b73*k3 + b74*k4 + b76*k6)
        
        # 5th order solution (used for error and next step if accepted)
        # Note: Standard DP uses a 7th stage for error estimation
        k7 = f(t + c7*h, y + h*(b71*k1 + b73*k3 + b74*k4 + b76*k6))
        calls += 1
        
        y_new5 = y + h*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k6 + 11/84*k7)
        
        # Error estimate
        err_vec = y_new5 - y_new4
        
        # Scale error
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new5))
        err_norm = np.max(np.abs(err_vec) / scale)
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new5 # Use 5th order for higher accuracy
            # Update step size
            if err_norm > 1e-15:
                h_new = 0.9 * h * (1.0 / err_norm) ** 0.2
            else:
                h_new = h * 5.0
            h = min(h_new, 5.0 * h)
        else:
            # Reject step, reduce h
            h_new = 0.9 * h * (1.0 / err_norm) ** 0.2
            h = max(h_new, 0.1 * h)
            
        # Safety checks
        if time.time() - start_time > 250:
            break
        if calls > 1900000:
            break
        if not np.all(np.isfinite(y)):
            break

    return y
