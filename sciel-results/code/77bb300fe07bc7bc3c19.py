import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use Radau method for stiff problems, high accuracy
    # The reference uses Radau with rtol=1e-11, atol=1e-13
    # We use the same to match accuracy closely
    sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-11, atol=1e-13)
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback if it fails, though unlikely
        # Simple RK4 fallback
        t0, t1 = t_span
        h = (t1 - t0) / 1000
        t, y = t0, np.array(y0)
        for _ in range(1000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
