"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Use a high-order explicit Runge-Kutta method with fixed steps.
    # We'll use a 5th-order Dormand-Prince method with a calibrated step size.
    
    # For the Lorenz system on [0, 2] with y0 = [1,1,1], we need high accuracy.
    # Let's use 20000 steps for the 5th-order method (7 calls per step = 140000 calls).
    # But that might be too many. Let's use a higher-order method or fewer steps.
    
    # Actually, let's use RK4 with a small step size. RK4 uses 4 calls per step.
    # With 20000 steps, that's 80000 calls.
    
    # Let's use a 5th-order method with fewer steps. DP5 uses 6 calls per step (k1-k6, k7 is for error estimation but we can skip it for fixed step).
    # Actually, DP5 as a 5th-order method uses 6 stages (k1-k6), with k7 for error estimation.
    # For a fixed-step 5th-order method, we can use 6 calls per step.
    
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    
    # Dormand-Prince 5(4) coefficients for the 5th-order solution
    # a21 = 1/5
    # a31 = 3/40, a32 = 9/40
    # a41 = 44/45, a42 = -56/15, a43 = 32/9
    # a51 = 19372/6561, a52 = -25360/2187, a53 = 64448/6561, a54 = -212/729
    # a61 = 9017/3168, a62 = -355/33, a63 = 46732/5247, a64 = 49/176, a65 = -5103/18656
    # b1 = 35/384, b3 = 0, b4 = 500/1113, b5 = 125/192, b6 = -2187/6784, b7 = 11/84
    # For the 5th-order solution, we use b1, b3, b4, b5, b6, b7 but b3=0 and we need k7.
    # Actually, the 5th-order solution uses k1, k4, k5, k6, k7 (k2, k3 are not in the 5th-order weights).
    # So we need 7 stages total, but k7 is computed from k1, k4, k5, k6.
    
    # Let's just use RK4 for simplicity and reliability.
    steps = 20000
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            # Fallback: return the last valid value or raise
            break
    
    return y
# EVOLVE-BLOCK-END
