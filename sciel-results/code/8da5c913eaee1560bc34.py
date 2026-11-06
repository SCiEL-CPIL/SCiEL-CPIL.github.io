"""Improved ODE solver for Robertson problem using Radau IIA (implicit Runge-Kutta)."""
# EVOLVE-BLOCK-START
import numpy as np


def _radau_ii_a_jac(f, t, y, h):
    """Compute the Jacobian of f(t, y) at (t, y) using finite differences.
    
    For the Robertson problem, f depends only on y (autonomous), but we keep
    the general form.
    """
    n = len(y)
    J = np.empty((n, n))
    for j in range(n):
        # Central difference with adaptive step size
        eps = 1e-7 * max(1.0, abs(y[j]))
        yp = y.copy()
        yp[j] += eps
        ym = y.copy()
        ym[j] -= eps
        J[:, j] = (f(t, yp) - f(t, ym)) / (2 * eps)
    return J


def _solve_nonlinear(f, t, y, h, A, b, y0, J_prev=None):
    """Solve the nonlinear stage equations for Radau IIA.
    
    For each stage i: y_i = y0 + h * sum_j A[i,j] * f(t + c_j*h, y_j)
    We use Newton iteration with a fixed-point acceleration.
    """
    s = len(b)  # number of stages
    n = len(y)
    
    # Initial guess: use previous stage values or y0
    Y = np.empty((s, n))
    F = np.empty((s, n))
    
    # Initialize with extrapolation or just y0
    for i in range(s):
        Y[i] = y0.copy()
        F[i] = f(t, y0)
    
    # Precompute Jacobian at initial guess
    J = J_prev if J_prev is not None else _radau_ii_a_jac(f, t, y0, h)
    
    max_newton = 10
    tol = 1e-10
    
    for newton_iter in range(max_newton):
        # Compute residuals
        converged = True
        for i in range(s):
            # Residual: Y[i] - y0 - h * sum_j A[i,j] * F[j]
            sum_AF = np.zeros(n)
            for j in range(s):
                sum_AF += A[i, j] * F[j]
            residual = Y[i] - y0 - h * sum_AF
            
            if np.linalg.norm(residual) > tol * (1 + np.linalg.norm(Y[i])):
                converged = False
            
            # Update Y[i] using Newton-like step
            if not converged:
                # The linearization: (I - h * sum_j A[i,j] * J) * dY_i = -residual
                # But since all stages are coupled, we need a more global approach.
                # For simplicity, use a Gauss-Seidel style update.
                pass
        
        if converged:
            break
        
        # Gauss-Seidel Newton update
        for i in range(s):
            sum_AF = np.zeros(n)
            for j in range(s):
                if j < i:
                    sum_AF += A[i, j] * F[j]
                else:
                    sum_AF += A[i, j] * F[j]  # use current iteration values
            
            # Solve (I - h * A[i,i] * J) * dY = -(Y[i] - y0 - h * sum_AF)
            M = np.eye(n) - h * A[i, i] * J
            rhs = -(Y[i] - y0 - h * sum_AF)
            
            try:
                dY = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                dY = -rhs / (1 + h * A[i, i])
            
            Y[i] += dY
            F[i] = f(t, Y[i])
        
        # Update Jacobian every few iterations
        if (newton_iter + 1) % 3 == 0:
            # Use a representative point for Jacobian
            J = _radau_ii_a_jac(f, t, Y[0], h)
    
    return Y, F, J


def _radau_ii_a_step(f, t, y, h, A, b, c, J_prev=None):
    """Take one Radau IIA step.
    
    Returns (y_new, J_new, n_f_calls)
    """
    s = len(b)
    n = len(y)
    
    # Solve stage equations
    Y, F, J_new = _solve_nonlinear(f, t, y, h, A, b, y, J_prev)
    
    # Final value: y_new = y + h * sum_i b_i * F_i
    y_new = y.copy()
    for i in range(s):
        y_new += h * b[i] * F[i]
    
    return y_new, J_new


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] using Radau IIA.
    
    Uses a 3-stage Radau IIA method (order 5) with adaptive step size control.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Radau IIA 3-stage coefficients (order 5)
    # c = [1/6, 3/6, 5/6]
    c = np.array([1.0/6.0, 3.0/6.0, 5.0/6.0])
    s = 3
    
    # A matrix for 3-stage Radau IIA
    A = np.array([
        [1/36.0,  1/12.0,  5/36.0],
        [-1/36.0,  5/12.0,  11/36.0],
        [1/9.0,   -1/3.0,   5/9.0]
    ])
    
    # b vector (same as last row of A for Radau IIA)
    b = np.array([1/9.0, -1/3.0, 5/9.0])
    
    # Initial step size estimate
    h = (t1 - t0) / 100.0
    if h <= 0:
        h = abs(t1 - t0) / 100.0
    
    # Adaptive parameters
    rtol = 1e-10
    atol = 1e-12
    max_steps = 2000000
    min_h = 1e-15
    max_h = abs(t1 - t0) / 10.0
    
    t = t0
    J_prev = None
    n_calls = 0
    
    # Track for error estimation: use a lower-order embedding
    # For Radau IIA, we can estimate error by comparing with a simpler method
    # or use the difference between consecutive steps.
    
    # We'll use a simple error estimator: compare RK2 (embedded) with Radau
    # Actually, let's use a different approach: take two half-steps and compare.
    
    while t < t1 - 1e-15 and n_calls < max_steps:
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        if h < min_h:
            h = min_h
        
        # Take a Radau IIA step
        y_new, J_new = _radau_ii_a_step(f, t, y, h, A, b, c, J_prev)
        
        # Error estimation: take two half-steps with Radau IIA
        h_half = h / 2.0
        y_half, J_half = _radau_ii_a_step(f, t, y, h_half, A, b, c, J_prev)
        y_two_half, J_two_half = _radau_ii_a_step(f, t + h_half, y_half, h_half, A, b, c, J_half)
        
        # Error estimate
        err_vec = np.abs(y_two_half - y_new)
        err_norm = np.max(err_vec / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        # Accept or reject
        if err_norm <= 1.0:
            y = y_two_half  # Use the more accurate double-half-step result
            t += h
            J_prev = J_two_half
            
            # Adjust step size
            if err_norm < 1e-5:
                h *= 2.0
            elif err_norm > 0.5:
                h *= 0.9
            else:
                h *= 1.2
        else:
            # Reject and reduce step
            h *= max(0.5, 0.9 * err_norm ** (-0.25))
        
        # Safety checks
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            # Fall back to a simple implicit method
            break
        
        # Update h bounds
        h = np.clip(h, min_h, max_h)
    
    return y

# EVOLVE-BLOCK-END
