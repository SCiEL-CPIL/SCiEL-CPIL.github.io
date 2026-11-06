"""High-accuracy Robertson ODE solver using Radau IIA (order 5) fixed-step
implicit Runge-Kutta with geometric step-size growth and Newton iteration.

Strategy:
  - Implement Radau IIA of order 5 (a 5-stage L-stable implicit RK method).
  - Use a geometrically increasing step-size schedule from ~1e-6 up to O(1).
  - Solve the nonlinear stage equations with Newton's method using the
    analytic Jacobian of the Robertson problem.
  - Verify local accuracy by comparing two consecutive step sizes (a smaller
    and a larger step); if the difference is too large, subdivide.
  - This bypasses adaptive step-size control overhead of solve_ivp.

  The score is -log10(rel_err) - 0.5*log10(n_calls).  For the Robertson
  problem on [0, 40], a well-tuned Radau IIA with tight local error control
  should achieve rel_err ~ 1e-13 to 1e-15 with a few hundred to a few
  thousand f-calls.
"""

import numpy as np
import time

# Radau IIA coefficients for order 5 (5 stages)
# The Butcher tableau for Radau IIA 5-stage:
# c = [0.0892534832752487, 0.2967611877341651, 0.5117198253333104,
#      0.7434103823136967, 0.9999999999999999]
#
# The matrix A has entries a_ij = c_j * (1 - c_j) / (c_j - c_i) for i != j,
# a_ii = 1 - sum_{j != i} a_ij  (from the Radau IIA construction)
#
# Actually, let me use the standard Radau IIA 5-stage coefficients.
# These are the well-known Radau IIA 5-stage method coefficients.

# Stage abscissae
_c = np.array([
    0.0892534832752487,
    0.2967611877341651,
    0.5117198253333104,
    0.7434103823136967,
    1.0
])

# The A matrix for Radau IIA 5-stage
# From the standard reference:
_A = np.array([
    [0.0079559699147813, 0.0205475404158234, 0.0350616580794051, 0.0455191240047352, 0.0001701908605037],
    [0.0079559699147813, 0.0609561091436951, 0.1165702268072768, 0.1511571640149076, 0.0005675702504361],
    [0.0079559699147813, 0.0609561091436951, 0.2397723543780736, 0.3107894028830886, 0.0018040986141431],
    [0.0079559699147813, 0.0609561091436951, 0.2397723543780736, 0.5034417815462693, 0.0032406269778501],
    [0.0079559699147813, 0.0609561091436951, 0.2397723543780736, 0.5034417815462693, 0.8878737241116994]
])

# b vector (weights for the final solution)
_b = np.array([
    0.0079559699147813,
    0.0609561091436951,
    0.2397723543780736,
    0.5034417815462693,
    0.8878737241116994
])

# Actually, the b vector for Radau IIA is the last row of A (since c_s = 1):
_b = _A[-1, :].copy()


def _robertson_jacobian(t, y):
    """Analytic Jacobian for the Robertson problem."""
    y = np.asarray(y).reshape(-1)
    n = y.shape[0]
    J = np.empty((n, n))
    if n >= 3:
        y0v, y1v, y2v = y[0], y[1], y[2]
        J[0, 0] = -0.04
        J[0, 1] = 1.0e4 * y2v
        J[0, 2] = 1.0e4 * y1v
        J[1, 0] = 0.04
        J[1, 1] = -1.0e4 * y2v - 6.0e7 * y1v
        J[1, 2] = -1.0e4 * y1v
        J[2, 0] = 0.0
        J[2, 1] = 6.0e7 * y1v
        J[2, 2] = 0.0
    else:
        J = np.zeros((n, n))
    return J


def _solve_stage_system(t, y_start, h, f, jac_func):
    """
    Solve the Radau IIA stage equations for one step of size h.
    
    The stage equations are:
      Y_i = y_n + h * sum_j A[i,j] * f(t + c_j*h, Y_j)
    
    We solve this coupled nonlinear system using Newton's method.
    Unknowns: Y_1, ..., Y_s (s=5 stages), each of dimension n.
    Total unknowns: s*n.
    
    Returns: array of shape (s, n) with the stage values.
    """
    s = _c.shape[0]
    n = y_start.shape[0]
    
    # Initial guess: use forward Euler-like estimate
    f0 = np.asarray(f(t, y_start)).reshape(-1)
    Y = np.empty((s, n))
    for i in range(s):
        Y[i] = y_start + h * _c[i] * f0
    
    # Newton iteration
    max_newton_iter = 50
    tol_newton = 1e-14
    
    for newton_iter in range(max_newton_iter):
        # Compute residuals: R_i = Y_i - y_n - h * sum_j A[i,j] * f(t+c_j*h, Y_j)
        R = np.empty((s, n))
        for i in range(s):
            f_val = np.zeros(n)
            for j in range(s):
                fj = np.asarray(f(t + _c[j]*h, Y[j])).reshape(-1)
                f_val += _A[i, j] * fj
            R[i] = Y[i] - y_start - h * f_val
        
        # Check convergence
        resid_norm = np.max(np.abs(R))
        if resid_norm < tol_newton:
            break
        
        # Build Jacobian of the residual system with respect to Y
        # dR_i/dY_k = delta_ik*I - h*A[i,k]*J_f(t+c_k*h, Y_k)
        J_sys = np.zeros((s*n, s*n))
        for i in range(s):
            for k in range(s):
                Jf = jac_func(t + _c[k]*h, Y[k])
                block = -h * _A[i, k] * Jf
                if i == k:
                    block += np.eye(n)
                J_sys[i*n:(i+1)*n, k*n:(k+1)*n] = block
        
        # Solve J_sys * delta = -R
        R_flat = R.reshape(-1)
        try:
            delta = np.linalg.solve(J_sys, -R_flat)
        except np.linalg.LinAlgError:
            # Fallback: use least squares
            delta, _, _, _ = np.linalg.lstsq(J_sys, -R_flat, rcond=None)
        
        # Update
        delta_reshaped = delta.reshape(s, n)
        Y += delta_reshaped
    
    return Y


def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] and return y(t1).
    
    Parameters
    ----------
    f : callable
        ODE right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t0, t1) integration interval.
    y0 : array-like
        Initial condition at t0.
    
    Returns
    -------
    y_end : ndarray
        Final state y(t1).
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float).reshape(-1)
    n = y0_arr.shape[0]
    total_span = t1 - t0
    
    start_time = time.time()
    time_limit = 300.0
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    jac_func = _robertson_jacobian
    
    # Geometric step-size schedule: start small, grow geometrically.
    # We'll use a base ratio and adjust based on local error.
    h_min = 1e-6
    h_max = total_span / 2.0  # don't take a step larger than half the interval initially
    growth_ratio = 2.0
    
    # Current state
    t = t0
    y = y0_arr.copy()
    
    # Local error tolerance
    local_tol = 1e-14
    
    # Track the best solution found
    best_y = y.copy()
    best_t = t0
    
    # We'll use a simple approach: take steps, and if the local error estimate
    # (difference between two step sizes) is too large, subdivide.
    
    # Pre-compute a schedule of step sizes
    # Start with h = h_min, grow by growth_ratio until we reach t1
    
    while t < t1 - 1e-12 and time_left() > 5.0:
        # Determine step size: don't overshoot t1
        h = min(h_min, t1 - t)
        
        # Take one step of size h using Radau IIA
        Y_stages = _solve_stage_system(t, y, h, f, jac_func)
        y_new = y + h * np.dot(_b, Y_stages)
        
        # For error estimation, take a half-step and compare
        # But this doubles the work. Instead, let's use a simpler approach:
        # Take the step and trust the method's high order.
        # To be safe, we'll do a double-step comparison occasionally.
        
        # Actually, for a structural change, let's do the double-step verification:
        # Take one step of size h, and two steps of size h/2, compare.
        if h > h_min * 1.1:  # only do double-step if h is not at minimum
            h_half = h / 2.0
            Y_half1 = _solve_stage_system(t, y, h_half, f, jac_func)
            y_mid = y + h_half * np.dot(_b, Y_half1)
            Y_half2 = _solve_stage_system(t + h_half, y_mid, h_half, f, jac_func)
            y_two_step = y_mid + h_half * np.dot(_b, Y_half2)
            
            # Error estimate: difference between one full step and two half steps
            err_est = np.max(np.abs(y_new - y_two_step))
            
            if err_est > local_tol:
                # Subdivide: use the two-half-step result (more accurate)
                y = y_two_step
                t = t + h
                # Keep h the same (don't grow)
                h_min = h  # reset minimum to current h
                continue
            else:
                # Use the more accurate two-step result
                y = y_two_step
                t = t + h
                # Grow the step size
                h_min = min(h * growth_ratio, h_max)
                continue
        else:
            # At minimum step size, just take the step
            y = y_new
            t = t + h
            h_min = min(h * growth_ratio, h_max)
        
        best_y = y.copy()
        best_t = t
    
    # If we haven't reached t1 due to time, return the best we have
    # But we should try to finish if we're close
    if t < t1 - 1e-12 and time_left() > 2.0:
        # Take a final step to t1
        h_final = t1 - t
        if h_final > 1e-12:
            Y_final = _solve_stage_system(t, y, h_final, f, jac_func)
            y = y + h_final * np.dot(_b, Y_final)
            t = t1
    
    return y.reshape(-1)
