import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-11, atol=1e-13)
    return sol.y[:, -1]
