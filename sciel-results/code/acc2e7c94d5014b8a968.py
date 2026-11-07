import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Strategy: Use a high-order fixed-step Runge-Kutta method (DOP853-like coefficients)
    # with enough steps to push the error below the 1e-14 floor, while keeping calls low.
    # DOP853 is 8th order.
    # Error ~ C * h^8.
    # If we use N steps, h = (t1-t0)/N.
    # We want error < 1e-14.
    # For Lorenz, constants are not huge.
    # Let's estimate.
    # If N=100, h=0.02. h^8 = 2.56e-14. Might be close to floor.
    # If N=50, h=0.04. h^8 = 6.5e-11. Error ~ 1e-10.
    # If N=200, h=0.01. h^8 = 1e-16. Error < 1e-14.
    # Calls = N * 11 (DOP853 has 11 stages? No, DOP853 has 12 stages, 11 if one is reused? 
    # Actually DOP853 has 12 stages, but the last one is the solution, so 11 evals? 
    # Let's check scipy DOP853. It has 12 stages. 
    # Wait, standard DOP853 has 12 stages.
    # Calls = N * 12.
    # If N=200, Calls = 2400.
    # Score = -log10(1e-14) - 0.5*log10(2400) = 14 - 0.5*3.38 = 14 - 1.69 = 12.31.
    # This is much better than 1.6.
    
    # Let's verify the number of stages for DOP853.
    # From scipy source: DOP853 has 12 stages.
    # Coefficients are available in scipy.integrate._ode or similar, but we can hardcode them 
    # or use a simpler high order method if coefficients are hard to get right without lookup.
    # However, implementing DOP853 manually is risky if I get a coefficient wrong.
    # Alternative: Use a known correct high-order method like RKF8(7) or similar?
    # Or just use scipy's solve_ivp with fixed step? No, solve_ivp is adaptive.
    # Can we force solve_ivp to use a fixed step? No.
    
    # Is there a fixed-step high-order RK in scipy? No.
    # We must implement it.
    # To avoid coefficient errors, I will implement a 5th order method (DP54) but with MANY more steps?
    # DP54 is 5th order. Error ~ h^5.
    # To get 1e-14, h^5 < 1e-14 => h < 1e-2.8 = 1.5e-3.
    # N > 2 / 0.0015 = 1333 steps.
    # Calls = N * 6 = 1333 * 6 = 8000.
    # Score = 14 - 0.5*log10(8000) = 14 - 1.7 = 12.3.
    # This is safer to implement correctly than DOP853 coefficients.
    # DP54 coefficients are standard.
    
    # Let's go with DP54 (Dormand-Prince 5(4)) fixed step.
    # It has 6 stages (5 for 5th order, 1 for error estimation, but we only use the 5th order update).
    # Actually, standard DP54 uses 6 function evaluations per step if we want error estimation, 
    # but if we just want the 5th order solution, we can use the first 5 stages? 
    # No, the 5th order solution uses 6 stages (b vector has 6 non-zero entries? 
    # Let's check DP54 b vector:
    # b = [1/5, 0, 3/10, 0, 0, 1/5] ? No.
    # Standard DP54:
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1]
    # b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    # This is a 6-stage method. The 5th order solution uses all 6 stages.
    # So 6 calls per step.
    
    # Let's use N=2000 steps to be safe on accuracy.
    # h = 2/2000 = 0.001.
    # h^5 = 1e-15.
    # Error should be well below 1e-14.
    # Calls = 2000 * 6 = 12000.
    # Score = 14 - 0.5*log10(12000) = 14 - 1.75 = 12.25.
    
    # Can we do better with DOP853?
    # If I can get DOP853 coefficients right, N=100, Calls=1200, Score=14-1.5=12.5.
    # Slightly better.
    # But risk of implementation error is high.
    # DP54 is very standard. I am confident in its coefficients.
    
    # Let's try to increase N for DP54 to ensure accuracy.
    # N=3000, Calls=18000. Score = 14 - 0.5*4.25 = 14 - 2.1 = 11.9.
    # N=2000 is a good balance.
    
    # Let's implement DP54.
    
    # Coefficients for DP54 (Dormand-Prince 5(4))
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    
    N = 2000
    h = (t1 - t0) / N
    t = t0
    y = y0.copy()
    
    for i in range(N):
        if time.time() > deadline - 5.0:
            # If we run out of time, return current y. 
            # But we are doing fixed steps, so we should finish quickly.
            # 2000 steps * 6 calls = 12000 calls. Should be < 1s.
            break
            
        k = np.zeros((6, n))
        
        for j in range(6):
            y_stage = y.copy()
            for m in range(j):
                y_stage += h * A[j][m] * k[m]
            k[j] = f(t + c[j]*h, y_stage)
        
        y = y + h * np.dot(b, k)
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
