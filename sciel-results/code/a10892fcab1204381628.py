import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using a fixed-step 5th-order Runge-Kutta method (Dormand-Prince).
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # DP5 Coefficients
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    A = [
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ]
    b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    
    N = 10000
    h = (t1 - t0) / N
    t = t0
    dim = len(y)
    
    for i in range(N):
        if time.time() > deadline - 5:
            break
            
        k = [np.zeros(dim) for _ in range(7)]
        
        for j in range(7):
            y_stage = y.copy()
            if j > 0:
                # Sum h * A[j][m] * k[m]
                # Using list comprehension or loop
                for m in range(j):
                    y_stage += h * A[j][m] * k[m]
            
            k[j] = f(t + c[j] * h, y_stage)
        
        # Update y
        y_new = y.copy()
        for j in range(7):
            y_new += h * b[j] * k[j]
        y = y_new
        t = t + h
        
    return y
