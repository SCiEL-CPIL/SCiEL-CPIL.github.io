"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using an adaptive high-order Runge-Kutta method.
    We use a 5(4) pair (Dormand-Prince style) with adaptive step sizing to achieve
    high accuracy with a controlled number of function calls.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard
    deadline = time.time() + 290.0  # 300s limit, 10s safety margin
    
    # Dormand-Prince 5(4) coefficients
    # This is a well-known RK method with embedded error estimator
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    c7 = 1.0
    
    a21 = 1/5
    a31 = 3/40
    a32 = 9/40
    a41 = 44/45
    a42 = -56/15
    a43 = 32/9
    a51 = 19372/6561
    a52 = -25360/2187
    a53 = 64448/6561
    a54 = -212/729
    a61 = 9017/3168
    a62 = -355/33
    a63 = 46732/5247
    a64 = 49/176
    a65 = -5103/18656
    
    b1 = 35/384
    b2 = 0
    b3 = 500/1113
    b4 = 125/192
    b5 = -2187/6784
    b6 = 11/84
    b7 = 0
    
    e1 = 71/57600
    e2 = 0
    e3 = -71/16695
    e4 = 71/1920
    e5 = -17253/339200
    e6 = 22/525
    e7 = -1/40
    
    def rk_step(t, y, h):
        """Perform one RK step and return (y_new, y_err)"""
        k1 = f(t, y)
        k2 = f(t + c2*h, y + h*(a21*k1))
        k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        k7 = f(t + c7*h, y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6))
        
        # 5th order solution
        y5 = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        # 4th order solution (for error estimate)
        y4 = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6 + b7*k7)
        
        # Actually, let me use the standard DP5 error estimate
        # The 4th order is: y4 = y + h*(71/57600*k1 + 0*k2 - 71/16695*k3 + 71/1920*k4 - 17253/339200*k5 + 22/525*k6 - 1/40*k7)
        y4 = y + h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7)
        
        return y5, y4 - y5
    
    # Adaptive stepping
    rtol = 1e-11
    atol = 1e-13
    h = (t1 - t0) / 100.0  # initial step
    t = t0
    max_steps = 50000
    min_h = 1e-10
    max_h = (t1 - t0) * 0.1
    
    for _ in range(max_steps):
        if t >= t1:
            break
        if time.time() > deadline:
            break
        
        # Limit step to not overshoot
        if t + h > t1:
            h = t1 - t
        
        y5, err = rk_step(t, y, h)
        
        # Check for non-finite values
        if not np.all(np.isfinite(y5)):
            h *= 0.5
            if h < min_h:
                return y  # Return best available
            continue
        
        # Error norm
        err_norm = np.max(np.abs(err) / (atol + rtol * np.abs(y5)))
        
        if err_norm <= 1.0:
            # Step accepted
            y = y5
            t += h
            # Adjust step size
            if err_norm < 1e-12:
                h *= 5.0
            else:
                h *= 0.9 * (1.0 / err_norm) ** 0.2
            h = min(max(h, min_h), max_h)
        else:
            # Step rejected, reduce h
            h *= 0.9 * (1.0 / err_norm) ** 0.2
            h = max(h, min_h)
    
    return y
# EVOLVE-BLOCK-END
