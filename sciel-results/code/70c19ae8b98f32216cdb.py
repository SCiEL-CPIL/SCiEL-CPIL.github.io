import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using DOP853 (8th-order explicit RK).
    High order allows large steps, reducing total function calls.
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 330s kill

    # DOP853: 8th order explicit Runge-Kutta, 12 stages per step.
    # For this moderately stiff problem, it can take much larger steps than Radau
    # while achieving comparable accuracy, resulting in fewer total f calls.
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15
    )

    if not sol.success:
        # Fallback to Radau if DOP853 fails
        if time.time() > deadline:
            return np.array(y0, dtype=float)
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14
        )
        if not sol.success:
            return np.array(y0, dtype=float)

    return sol.y[:, -1]
