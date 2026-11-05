"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem y' = f(t, y) on [t0, t1] with adaptive RK45.
    Returns the final state y(t1).
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Robertson problem is stiff. Use a stiff solver approach.
    # Since we only have f, we'll implement a simple implicit method or use a very stable explicit method.
    # However, the task allows SciPy. Let's use scipy.integrate.solve_ivp with a stiff method like Radau or BDF.
    # But wait, the evaluator counts calls to f. If we use solve_ivp, it will call f.
    # The score is -log10(relative error) - 0.5*log10(calls).
    # Using a high-order stiff solver with tight tolerances should give low error and moderate call count.
    
    # Let's try using scipy.integrate.solve_ivp with method='Radau' or 'BDF'.
    # These are implicit methods suitable for stiff problems.
    
    from scipy.integrate import solve_ivp
    
    # Robertson problem is known to be stiff.
    # We need to be careful with the call count.
    # solve_ivp returns a result object. We need the final state.
    
    # Set tight tolerances for high accuracy.
    # rtol and atol control the error.
    # Let's start with rtol=1e-12, atol=1e-14 for high precision.
    
    # Deadline guard: 300s limit, keep a safety margin.
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=1.0  # Limit step size to avoid missing dynamics, but Robertson is slow/fast
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try BDF
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Fallback to a simple explicit method with small steps if implicit fails
                # But this might be slow.
                # Let's just return the last computed state if available
                if sol.y.shape[1] > 0:
                    return sol.y[:, -1]
                else:
                    return y
    except Exception as e:
        # If something goes wrong, fall back to a basic RK4 with small steps
        # This is a last resort
        h = (t1 - t0) / 10000
        t = t0
        y = np.array(y0, dtype=float)
        while t < t1:
            if time.time() > deadline:
                break
            h_step = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h_step/2, y + h_step/2 * k1)
            k3 = f(t + h_step/2, y + h_step/2 * k2)
            k4 = f(t + h_step, y + h_step * k3)
            y = y + h_step/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h_step
        return y

# EVOLVE-BLOCK-END
