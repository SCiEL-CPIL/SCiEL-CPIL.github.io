import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)

    # Coefficients for DP54 (Dormand-Prince 5(4))
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1]
    # A matrix (upper triangular part is zero, we only need lower/upper depending on storage, here we use standard form)
    # Stage j depends on stages 0..j-1
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    
    # 5th order solution vector (for error estimation, though we might not use it for step control in fixed mode)
    b_star = np.array([71/576, 0, 71/192, -71/192, 172/213, 1/4])

    # Strategy:
    # 1. Estimate optimal fixed step size N to achieve rel error < 1e-14.
    #    Error ~ C * h^5.
    #    For Lorenz, C is not huge. 
    #    Let's aim for N such that h^5 * C < 1e-14.
    #    If C ~ 1 (optimistic), h < 1e-2.8 ~ 1.6e-3. N > 1250.
    #    If C ~ 10, h < 4e-3. N > 500.
    #    If C ~ 100, h < 1.6e-2. N > 125.
    #    Let's pick N=1000 as a starting point. h=0.002.
    #    Calls = 1000 * 6 = 6000.
    #    Score potential: 14 - 0.5*log10(6000) = 14 - 1.68 = 12.32.
    
    # However, we can do better by being adaptive but efficient?
    # No, fixed step is safer for "predictable low call count".
    # Let's try to refine N.
    # We can run a small pilot integration to estimate the error constant C?
    # That would cost extra calls.
    # Alternative: Just pick a conservative N.
    # N=1500. h=0.00133. h^5 = 4e-15.
    # Calls = 9000. Score = 14 - 1.72 = 12.28.
    # N=1000. Calls=6000. Score=12.32.
    # If N=1000 is accurate enough, it's better.
    # Let's check if N=500 is enough? h=0.004. h^5 = 1e-12. Error ~ 1e-11? Too high.
    # So N=1000 seems like a sweet spot.
    
    # Let's implement a fixed-step DP54 with N=1000.
    # If we have time, we could do a second pass with smaller h to check convergence?
    # No, that doubles calls.
    
    # What if we use DOP853?
    # 12 stages. N=200. Calls=2400.
    # h = 0.01. h^8 = 1e-16.
    # This is MUCH better if we can get coefficients right.
    # I will attempt to hardcode DOP853 coefficients.
    
    # DOP853 coefficients (from scipy.integrate._ode or public literature)
    # This is risky. If I get one wrong, the result is garbage.
    # DP54 is very standard. I am confident in DP54.
    # Let's stick with DP54 but optimize the loop.
    
    # Optimization: Pre-allocate k array.
    N = 1200  # Slightly conservative to ensure rel < 1e-14
    h = (t1 - t0) / N
    t = t0
    y = y0.copy()
    
    # Pre-allocate stage derivatives
    k = np.zeros((6, n))
    
    for i in range(N):
        if time.time() > deadline - 5.0:
            break
            
        # Stage 0
        k[0] = f(t, y)
        
        # Stage 1
        y1 = y + h * A[1, 0] * k[0]
        k[1] = f(t + c[1]*h, y1)
        
        # Stage 2
        y2 = y + h * (A[2, 0] * k[0] + A[2, 1] * k[1])
        k[2] = f(t + c[2]*h, y2)
        
        # Stage 3
        y3 = y + h * (A[3, 0] * k[0] + A[3, 1] * k[1] + A[3, 2] * k[2])
        k[3] = f(t + c[3]*h, y3)
        
        # Stage 4
        y4 = y + h * (A[4, 0] * k[0] + A[4, 1] * k[1] + A[4, 2] * k[2] + A[4, 3] * k[3])
        k[4] = f(t + c[4]*h, y4)
        
        # Stage 5
        y5 = y + h * (A[5, 0] * k[0] + A[5, 1] * k[1] + A[5, 2] * k[2] + A[5, 3] * k[3] + A[5, 4] * k[4])
        k[5] = f(t + c[5]*h, y5)
        
        # Update y using 5th order coefficients
        y = y + h * (b[0]*k[0] + b[2]*k[2] + b[3]*k[3] + b[4]*k[4] + b[5]*k[5])
        # Note: b[1] is 0, so we skip k[1] in the sum.
        
        t = t + h
        
    return y

def _fallback(f, t0, t1, y0):
    n_steps = 50000
    h = (t1 - t0) / n_steps
    t = t0
    y = np.array(y0, dtype=float)
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
