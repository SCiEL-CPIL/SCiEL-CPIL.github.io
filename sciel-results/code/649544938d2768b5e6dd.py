import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Dormand-Prince 5(4) coefficients
    # c2=1/5, c3=3/10, c4=4/5, c5=8/9, c6=1
    # a21=1/5
    # a31=3/40, a32=9/40
    # a41=44/45, a42=-56/15, a43=32/9
    # a51=19372/6561, a52=-25360/2187, a53=64448/6561, a54=-212/729
    # a61=9017/3168, a62=-355/33, a63=46732/5247, a64=49/176, a65=-5103/18656
    # b5: 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    # b4: 51/720, 0, 3/40, 1/5, -17/60, 0
    # e = b5 - b4: 71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525
    
    # Precompute coefficients
    a21 = 1/5
    a31 = 3/40; a32 = 9/40
    a41 = 44/45; a42 = -56/15; a43 = 32/9
    a51 = 19372/6561; a52 = -25360/2187; a53 = 64448/6561; a54 = -212/729
    a61 = 9017/3168; a62 = -355/33; a63 = 46732/5247; a64 = 49/176; a65 = -5103/18656
    
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    b4 = np.array([51/720, 0, 3/40, 1/5, -17/60, 0])
    e = b5 - b4
    
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    
    rtol = 1e-10
    atol = 1e-12
    max_steps = 10000
    min_step = 1e-12
    max_step = (t1 - t0) / 10
    
    t = t0
    h = max_step / 10.0 # Initial step estimate
    
    # FSAL: First stage of next step is last stage of current
    k_prev = f(t, y)
    calls = 1
    
    start_time = time.time()
    deadline = start_time + 280.0 # 20s margin
    
    while t < t1:
        if time.time() > deadline:
            break
            
        if t + h > t1:
            h = t1 - t
        if h < min_step:
            h = min_step
            
        # Stage 1
        k1 = k_prev
        
        # Stage 2
        y2 = y + h * a21 * k1
        k2 = f(t + c[1]*h, y2)
        calls += 1
        
        # Stage 3
        y3 = y + h * (a31*k1 + a32*k2)
        k3 = f(t + c[2]*h, y3)
        calls += 1
        
        # Stage 4
        y4 = y + h * (a41*k1 + a42*k2 + a43*k3)
        k4 = f(t + c[3]*h, y4)
        calls += 1
        
        # Stage 5
        y5 = y + h * (a51*k1 + a52*k2 + a53*k3 + a54*k4)
        k5 = f(t + c[4]*h, y5)
        calls += 1
        
        # Stage 6
        y6 = y + h * (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)
        k6 = f(t + c[5]*h, y6)
        calls += 1
        
        # Error estimate
        y_err = h * (e[0]*k1 + e[1]*k2 + e[2]*k3 + e[3]*k4 + e[4]*k5 + e[5]*k6)
        err_norm = np.linalg.norm(y_err / (atol + rtol * np.maximum(np.abs(y), np.abs(y + h*b5))))
        
        if err_norm <= 1.0:
            # Accept step
            y_new = y + h * (b5[0]*k1 + b5[1]*k2 + b5[2]*k3 + b5[3]*k4 + b5[4]*k5 + b5[5]*k6)
            t_new = t + h
            
            # Adjust step size for next step
            if err_norm > 0:
                factor = 0.9 * (1.0 / err_norm) ** 0.2
            else:
                factor = 1.1
            h = h * factor
            if h > max_step:
                h = max_step
            if h < min_step:
                h = min_step
                
            y = y_new
            t = t_new
            k_prev = k6 # FSAL
        else:
            # Reject step
            factor = 0.9 * (1.0 / err_norm) ** 0.25
            if factor < 0.1:
                factor = 0.1
            h = h * factor
            if h < min_step:
                h = min_step
                
        if calls > 1900000:
            break

    return y
