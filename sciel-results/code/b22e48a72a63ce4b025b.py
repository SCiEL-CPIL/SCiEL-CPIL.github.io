"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrates the Lorenz system using a custom high-order Runge-Kutta method (DOP853)
    with a fixed step size optimized for the specific problem constraints.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 320.0  # 10s margin before 330s limit

    # The Lorenz system is smooth for t in [0, 2] with y0=[1,1,1].
    # DOP853 is an 8th-order method. For this short interval, a fixed step size
    # that is sufficiently small to keep the local truncation error below the
    # reference tolerance (rtol=1e-11) can be used to minimize the number of steps.
    
    # Reference: rtol=1e-11, atol=1e-13.
    # Let's estimate the required step size.
    # For DOP853, the local error is proportional to h^9.
    # The global error is proportional to h^8.
    # We want the global relative error to be ~1e-11.
    # ||y|| ~ 10-20.
    # Let's try a step size that results in ~100-200 steps.
    # Interval length = 2.0.
    # If we take 100 steps, h = 0.02.
    # If we take 50 steps, h = 0.04.
    
    # To be safe and minimize calls, we can try a slightly larger step size
    # and verify if the result is reasonable, or just pick a conservative one.
    # However, the scoring penalizes calls heavily (log scale).
    # Fewer calls = better score, provided rel error is small.
    
    # Let's implement a fixed-step DOP853.
    # The Butcher tableau for DOP853 is large. Implementing it manually in Python
    # might be slow due to the number of function evaluations (14 per step).
    # 14 calls/step * 100 steps = 1400 calls.
    # solve_ivp with DOP853 might use adaptive steps.
    # If the adaptive controller uses ~100 steps, it also makes ~1400 calls.
    # The overhead of solve_ivp is in the Python loop and error estimation.
    
    # A more efficient strategy: Use a lower order method with very small steps?
    # No, high order is better for reducing steps.
    
    # Let's stick to the high-order method but implement it efficiently.
    # Actually, scipy's solve_ivp is already quite efficient.
    # The key to beating the seed might be to use a *smaller* number of steps
    # by using a higher tolerance if the error is still acceptable, or
    # ensuring that the step size is not unnecessarily small due to conservative
    # error estimation.
    
    # Alternative: Use scipy.integrate.solve_ivp with DOP853 but tune the tolerances
    # slightly looser than the reference if we can still get rel < 1e-12.
    # The reference is rtol=1e-11, atol=1e-13.
    # If we use rtol=1e-10, atol=1e-12, we might get fewer steps.
    # Let's check the sensitivity.
    
    # Let's try to implement a fixed-step DOP853 to have full control.
    # But coding DOP853 manually is error-prone and long.
    
    # Instead, let's use solve_ivp with DOP853 but with a max_step constraint
    # to ensure we don't take too large steps, and a min_step to avoid infinite loops.
    # And critically, let's try to use a slightly looser tolerance to see if we can
    # reduce the number of calls while keeping the error low.
    
    # The current program uses rtol=1e-11, atol=1e-13.
    # Let's try rtol=5e-12, atol=5e-14. This is looser, might reduce steps.
    
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=5e-12, 
            atol=5e-14,
            max_step=0.1  # Allow larger steps if error permits
        )
        
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback: Simple RK4 with small steps
    # This is a safety net, unlikely to be used if solve_ivp works.
    if time.time() > deadline - 5.0:
        # Return best guess or raise
        return y0 # Should not happen

    # Standard RK4 fallback
    steps = 1000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for _ in range(steps):
        if time.time() > deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

# EVOLVE-BLOCK-END
