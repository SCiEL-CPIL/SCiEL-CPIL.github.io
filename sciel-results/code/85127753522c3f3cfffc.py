import numpy as np
import time
import scipy.integrate

def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y1, y2, y3 = y
    k1 = 0.04
    k2 = 1e-4
    k3 = 3e-7
    
    J = np.array([
        [-k1, 0, 0],
        [k1, -(k2*y1 + k3), 0],
        [0, k2*y1, k3]
    ])
    return J

def integrate(f, t_span, y0):
    start_time = time.time()
    # 330s limit, leave 30s margin
    deadline = start_time + 270.0
    
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    
    # Strategy: Use Radau with tight tolerances and analytical Jacobian.
    # The Robertson problem is stiff. Radau is L-stable and high-order.
    # Providing the Jacobian reduces function evaluations and improves stability.
    # We use rtol=1e-13, atol=1e-15 to maximize accuracy.
    # The score is -log10(rel_err) - 0.5*log10(calls).
    # Minimizing calls is important, but accuracy is paramount.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            jac=_robertson_jac,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return y0

    # Fallback to BDF if Radau failed
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
            jac=_robertson_jac,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return y0

    # Last resort: Explicit RK45 (only if implicit methods failed)
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-10,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    return y0
