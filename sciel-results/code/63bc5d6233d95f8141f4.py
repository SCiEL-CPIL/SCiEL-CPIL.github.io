"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use Radau for stiff problems like Robertson
    # Tight tolerances for high accuracy score
    sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-9, atol=1e-12)
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to BDF if Radau fails
        sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-9, atol=1e-12)
        return sol.y[:, -1]
# EVOLVE-BLOCK-END
