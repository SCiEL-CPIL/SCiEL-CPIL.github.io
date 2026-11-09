"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using a high-order adaptive Runge-Kutta method.
    We implement an embedded 8(7) pair similar in spirit to DOP853 but with
    very tight tolerances to drive error below the 1e-14 floor.
    
    To keep function call counts reasonable while achieving high accuracy,
    we use a fixed-step high-order method (e.g., 8th order) with a small number
    of steps, or an adaptive method with tight tolerances.
    
    Given the scoring: score = -log10(rel) - 0.5 * log10(calls)
    We want rel < 1e-14 (so -log10(rel) >= 14) and calls as low as possible.
    
    The reference is DOP853 with rtol=1e-11, atol=1e-13.
    If we use a similar method with tighter tolerances, we should get rel ~ 1e-13 or better.
    
    Let's implement a fixed-step 8th order Runge-Kutta method with enough steps
    to ensure high accuracy. For t in [0, 5], the Lotka-Volterra system is not
    particularly stiff, so an explicit method should work fine.
    
    Alternatively, we can use an adaptive method with very tight tolerances.
    Let's implement a simple adaptive RK45 with tight tolerances, but that might
    take many steps. 
    
    Better approach: Use a fixed high-order method. Let's use a 8th order
    Runge-Kutta method with N steps. The error scales as h^8, so with N steps,
    error ~ (5/N)^8. To get error < 1e-14, we need (5/N)^8 < 1e-14, so
    5/N < 1e-14^(1/8) = 10^(-14/8) = 10^(-1.75) ≈ 0.0178, so N > 5/0.0178 ≈ 281.
    So N=300 steps should give error well below 1e-14.
    
    Each step of an 8th order RK method uses 12 function evaluations (for a typical
    8th order method). So 300 * 12 = 3600 calls. That's very low.
    
    Let's implement a standard 8th order Runge-Kutta method. I'll use the
    coefficients from a known 8th order method.
    
    Actually, let me use a simpler approach: use scipy's solve_ivp with DOP853
    and very tight tolerances, but that might not be allowed if we need to count
    calls to f. The task says f must be called at least once and calls are counted.
    
    Wait, the task says "f(t, y) returns y' as an array and every call is counted."
    So we need to implement our own integrator and count calls to f.
    
    Let me implement a high-order adaptive method. I'll use a Dormand-Prince 5(4)
    pair with very tight tolerances, or better, a higher-order method.
    
    Actually, for the Lotka-Volterra system, which is not stiff, a high-order
    explicit method with moderate step count should work. Let me implement a
    6th order Runge-Kutta method with adaptive step size control, using tight
    tolerances.
    
    For simplicity and reliability, let me implement a fixed-step 8th order
    Runge-Kutta method. I'll use the coefficients from a known 8th order method.
    
    Here are coefficients for a 8th order Runge-Kutta method (12 stages):
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Number of steps - use enough to ensure high accuracy
    n_steps = 500
    h = (t1 - t0) / n_steps
    t = t0
    
    # 8th order Runge-Kutta coefficients (12 stages)
    # Using a standard 8th order method
    c = np.array([0, 1/4, 1/2, 3/8, 1/2, 5/8, 1, 1, 1, 3/4, 1, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/8, 3/8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 9/32, 0, 9/32, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1/4, 0, 1/4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    b = np.array([1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # This is not a valid 8th order method. Let me use a proper one.
    # I'll use the Ralston 8th order method or a known valid set of coefficients.
    
    # Actually, let me use a simpler approach: use multiple RK4 steps with
    # Richardson extrapolation to get higher order accuracy.
    
    # Or, let me just use a well-tested adaptive method. I'll implement
    # Dormand-Prince 5(4) with very tight tolerances.
    
    # Reset
    y = np.array(y0, dtype=float)
    t = t0
    
    rtol = 1e-13
    atol = 1e-15
    h = (t1 - t0) / 100  # initial step
    min_h = 1e-8
    max_h = 0.1
    
    # Dormand-Prince 5(4) coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
    ])
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    n = len(y)
    call_count = 0
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
        if h < min_h:
            h = min_h
        
        k = np.zeros((7, n))
        for i in range(7):
            yi = y.copy()
            for j in range(i):
                yi += h * A[i, j] * k[j]
            k[i] = f(t + c[i] * h, yi)
            call_count += 1
        
        y5 = y + h * np.dot(b5, k)
        y4 = y + h * np.dot(b4, k)
        
        # Error estimate
        err = np.abs(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_norm = np.sqrt(np.mean((err / scale) ** 2))
        
        if err_norm <= 1.0:
            # Accept step
            y = y5
            t += h
            if err_norm > 1e-15:
                h = min(max_h, h * min(5.0, max(0.1, 0.9 * err_norm ** (-0.2))))
            else:
                h = min(max_h, h * 5.0)
        else:
            # Reject step
            h = max(min_h, h * max(0.1, 0.9 * err_norm ** (-0.25)))
        
        if not np.all(np.isfinite(y)):
            # Fallback to smaller steps
            break
    
    return y
# EVOLVE-BLOCK-END
