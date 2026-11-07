import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Dormand-Prince RK45 coefficients
    c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    
    b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
    e1, e2, e3, e4, e5, e6, e7 = 71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40
    
    rtol = 1e-11
    atol = 1e-13
    h = 0.01 # Initial step
    
    calls = 0
    
    def eval_f(t, y):
        nonlocal calls
        calls += 1
        return np.array(f(t, y), dtype=float)
        
    while t < t1:
        if t + h > t1:
            h = t1 - t
            
        k1 = eval_f(t, y)
        k2 = eval_f(t + c2*h, y + h*a21*k1)
        k3 = eval_f(t + c3*h, y + h*(a31*k1 + a32*k2))
        k4 = eval_f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        k5 = eval_f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = eval_f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        
        # 4th order solution
        y_new = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        
        # 5th order solution for error estimation
        k7 = eval_f(t + c7*h, y + h*(a71*k1 + a73*k3 + a74*k4 + a75*k5 + a76*k6))
        y_new_high = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6 + b7*k7)
        
        # Error estimation
        err = np.linalg.norm((y_new_high - y_new) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err <= 1.0:
            t = t + h
            y = y_new_high # Use higher order result
            
            # Adjust step size
            if err == 0:
                h = h * 5.0
            else:
                h = h * (0.9 * (1.0/err))**(1/5)
            
            # Limit step size increase
            if h > 0.1: h = 0.1
            if h < 1e-6: h = 1e-6
        else:
            # Reject step, reduce h
            h = h * max(0.1, 0.9 * (1.0/err)**(1/5))
            
        if calls > 1500000:
            break
            
    return y
