"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using a high-accuracy
    combination of DOP853 and Radau solvers with tight tolerances.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Very tight tolerances to minimize truncation error
    rtol = 1e-13
    atol = 1e-15
    
    original_f = f
    
    # Solve with DOP853 (explicit, order 8)
    sol_dop853 = solve_ivp(
        original_f,
        [t0, t1],
        y0,
        method='DOP853',
        rtol=rtol,
        atol=atol
    )
    
    # Solve with Radau (implicit, order 5, good for stiff)
    sol_radau = solve_ivp(
        original_f,
        [t0, t1],
        y0,
        method='Radau',
        rtol=rtol,
        atol=atol
    )
    
    # Fallback logic if solvers fail
    if not sol_dop853.success and not sol_radau.success:
        # Fallback to RK4 with fixed steps if both fail (unlikely)
        h = (t1 - t0) / 10000
        t, y = t0, y0.copy()
        for _ in range(10000):
            k1 = original_f(t, y)
            k2 = original_f(t + h / 2, y + h / 2 * k1)
            k3 = original_f(t + h / 2, y + h / 2 * k2)
            k4 = original_f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    if not sol_dop853.success:
        return sol_radau.y[:, -1]
    
    if not sol_radau.success:
        return sol_dop853.y[:, -1]
    
    y_dop853 = sol_dop853.y[:, -1]
    y_radau = sol_radau.y[:, -1]
    
    # Compare solutions to assess stability/stiffness
    diff = np.max(np.abs(y_dop853 - y_radau))
    scale = np.max(np.abs(y_dop853)) + 1e-14
    rel_diff = diff / scale
    
    # If the solutions agree closely, DOP853 (higher order) is likely more accurate.
    # If they disagree significantly, the problem might be stiff, and Radau (implicit)
    # might be more stable, or DOP853 might have taken too small steps and accumulated
    # more round-off? Actually, DOP853 is very stable for non-stiff.
    # For convection-diffusion, if it's not extremely stiff, DOP853 is preferred.
    
    # Heuristic: If relative difference is small, use DOP853.
    # If relative difference is large, use a weighted average or Radau?
    # Given the score function, we want minimum error.
    # DOP853 is order 8, Radau is order 5.
    # Unless stiffness forces DOP853 to take extremely small steps (increasing round-off),
    # DOP853 should be more accurate.
    
    if rel_diff < 1e-10:
        return y_dop853
    else:
        # If they differ, it's hard to say which is better without a reference.
        # However, for standard convection-diffusion, DOP853 is usually superior.
        # We'll stick with DOP853 but perhaps blend if the difference is moderate?
        # Blending can introduce error if one is significantly worse.
        # Let's return DOP853.
        return y_dop853
# EVOLVE-BLOCK-END
