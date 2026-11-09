"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using a high-order explicit Runge-Kutta method.
    We use a 5(4) pair (Dormand-Prince style) with adaptive step size control,
    or simply a fixed-step high-order RK method.
    
    Given the constraints:
    - f is called at least once
    - More than 2,000,000 calls is invalid
    - Score = -log10(rel) - 0.5 * log10(calls)
    
    The reference is DOP853 with rtol=1e-11, atol=1e-13.
    
    Strategy: Use a high-order fixed-step Runge-Kutta method. 
    Since the Lorenz system is chaotic but not stiff over [0,2], an explicit high-order method should work well.
    
    Let's use a 6th-order Runge-Kutta method with a reasonable number of steps.
    With 1000 steps, that's 6000 calls (6 stages per step). That's well under 2M.
    
    Actually, let's use a simpler approach: RK45 with adaptive stepping, but we need to count calls.
    
    Alternatively, let's use a fixed-step high-order method. Let's use RK4 with many steps.
    With 10000 steps, that's 40000 calls. 
    -log10(rel) for RK4 with 10000 steps over [0,2]: h=0.0002, local error ~ h^5, global ~ h^4 = 1.6e-15.
    That should be very accurate.
    
    But wait, the Lorenz system is chaotic, so errors grow exponentially. The Lyapunov exponent is ~0.9.
    Over t=2, errors grow by exp(0.9*2) ~ 6.4x. So global error might be ~ 1e-14 or so.
    
    Let's use a higher order method. Let's implement a 6th-order RK method.
    
    Actually, let's use scipy's solve_ivp with DOP853 directly? No, we need to call f ourselves and count calls.
    
    Let's implement a high-order RK method. I'll use a 6th-order method with 2000 steps.
    That's 12000 calls.
    
    Let me implement a classical 6th-order RK method.
    
    Alternatively, I can use a simpler approach: use RK4 with adaptive step size, monitoring the error.
    
    Let's go with a fixed-step 6th-order RK method. Here's a 6th-order RK method (from Hairer, Norsett, Wanner):
    
    Actually, let me use a well-known 6th-order method. I'll use the one from Butcher's table.
    
    For simplicity and reliability, let me use RK4 with a very small step size.
    With 5000 steps: h = 0.0004, 20000 calls.
    Global error ~ h^4 = 2.56e-14, times chaos growth ~ 6.4, so ~1.6e-13.
    That might not be enough for -log10(rel) to be high.
    
    Let's use 10000 steps: h = 0.0002, 40000 calls.
    Global error ~ h^4 = 1.6e-15, times 6.4 ~ 1e-14.
    -log10(1e-14) = 14. -0.5*log10(40000) = -0.5*4.6 = -2.3. Score ~ 11.7.
    
    With a 6th-order method and 2000 steps: h = 0.001, 12000 calls.
    Global error ~ h^6 = 1e-18, times 6.4 ~ 6.4e-18.
    -log10(6.4e-18) ~ 17.2. -0.5*log10(12000) = -0.5*4.08 = -2.04. Score ~ 15.2.
    
    That's much better. Let me implement a 6th-order RK method.
    
    I'll use the following 6th-order RK method (6 stages):
    
    c = [0, 1/5, 3/10, 4/5, 8/9, 1]
    a = [[0,0,0,0,0,0],
         [1/5,0,0,0,0,0],
         [3/40,9/40,0,0,0,0],
         [44/45,-56/15,32/9,0,0,0],
         [-81/40,36/5,-84/5,95/9,0,0],
         [36/125,0,0,200/625,210/625,0]]  # This is not quite right
    
    Let me use a known 6th-order method. I'll use the one from Hairer & Wanner.
    
    Actually, for reliability, let me just use a high number of RK4 steps.
    With 20000 steps: 80000 calls.
    Global error ~ (1e-4)^4 = 1e-16, times 6.4 ~ 6.4e-16.
    -log10(6.4e-16) ~ 15.2. -0.5*log10(80000) = -0.5*4.9 = -2.45. Score ~ 12.7.
    
    That's worse than the 6th-order method. Let me implement a proper 6th-order RK.
    
    I'll use the following 6th-order RK method (from Butcher):
    
    c1=0, c2=1/5, c3=3/10, c4=4/5, c5=8/9, c6=1
    a21=1/5
    a31=3/40, a32=9/40
    a41=44/45, a42=-56/15, a43=32/9
    a51=-81/40, a52=36/5, a53=-84/5, a54=95/9
    a61=36/125, a62=0, a63=0, a64=200/625, a65=210/625
    b1=36/125, b2=0, b3=0, b4=200/625, b5=210/625, b6=0
    
    Wait, that's not a valid 6th-order method. Let me look up a correct one.
    
    For simplicity and correctness, let me use a 5th-order method (Dormand-Prince 5(4) with the 5th order coefficients) with adaptive stepping, but that's complex.
    
    Let me just use RK4 with 10000 steps. It's simple and reliable.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
