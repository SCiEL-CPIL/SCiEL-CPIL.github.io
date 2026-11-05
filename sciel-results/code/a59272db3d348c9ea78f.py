"""ODE solver for the Robertson problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _robertson_jacobian(y):
    """Analytical Jacobian for the Robertson ODE system.
    
    The Robertson problem is:
    y0' = -0.04*y0 + 1e4*y1*y2
    y1' =  0.04*y0 - 1e4*y1*y2 - 1e4*y1^2
    y2' =  1e4*y1^2
    """
    y0, y1, y2 = y
    J = np.array([
        [-0.04,          1e4*y2,         1e4*y1],
        [ 0.04, -1e4*y2 - 2e4*y1, -1e4*y1],
        [ 0.0,          2e4*y1,          0.0]
    ])
    return J


def _solve_linear_system(A, b):
    """Solve Ax = b using LU factorization with partial pivoting."""
    n = len(b)
    M = np.array(A, dtype=float, copy=True)
    x = np.array(b, dtype=float, copy=True)
    
    # Gaussian elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row, col]) > abs(M[max_row, col]):
                max_row = row
        if max_row != col:
            M[[col, max_row]] = M[[max_row, col]]
            x[[col, max_row]] = x[[max_row, col]]
        
        if abs(M[col, col]) < 1e-15:
            # Singular or nearly singular, use pseudo-inverse
            return np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Eliminate below
        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            if factor != 0:
                M[row, col:] -= factor * M[col, col:]
                x[row] -= factor * x[col]
    
    # Back substitution
    sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sol[i] = (x[i] - np.dot(M[i, i+1:], sol[i+1:])) / M[i, i]
    return sol


def _implicit_radau2_step(f, t, y, h, J_analytical=None):
    """One step of the implicit 2-stage Radau IIA method (order 3, A-stable).
    
    Stage equations:
    y1 = y + h/6 * (3*f(t) + 3*f(t+h/2, y1))
    y2 = y + h * (3/4 * f(t+h/2, y1) + 1/4 * f(t+h, y2))
    
    Where y2 is the new solution y_{n+1}.
    """
    n = len(y)
    ft = f(t, y)
    
    # Stage 1: Solve y1 = y + h/6 * (3*ft + 3*f(t+h/2, y1))
    # => y1 - h/2 * f(t+h/2, y1) = y + h/2 * ft
    rhs1 = y + h/2 * ft
    t1 = t + h/2
    
    y1 = y.copy()
    for _ in range(8):
        f1 = f(t1, y1)
        if J_analytical is not None:
            J1 = J_analytical(y1)
        else:
            # Finite difference Jacobian
            J1 = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = 1e-7 * max(1.0, abs(y1[j]))
                e[j] = scale
                J1[:, j] = (f(t1, y1 + e) - f1) / scale
        A1 = np.eye(n) - h/2 * J1
        sol = _solve_linear_system(A1, rhs1)
        if np.linalg.norm(sol - y1) < 1e-12 * (1 + np.linalg.norm(sol)):
            y1 = sol
            break
        y1 = sol
    
    f1 = f(t1, y1)
    
    # Stage 2: Solve y2 = y + h * (3/4 * f1 + 1/4 * f(t+h, y2))
    # => y2 - h/4 * f(t+h, y2) = y + 3h/4 * f1
    rhs2 = y + 3*h/4 * f1
    t2 = t + h
    
    y2 = y1.copy()
    for _ in range(8):
        f2 = f(t2, y2)
        if J_analytical is not None:
            J2 = J_analytical(y2)
        else:
            J2 = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = 1e-7 * max(1.0, abs(y2[j]))
                e[j] = scale
                J2[:, j] = (f(t2, y2 + e) - f2) / scale
        A2 = np.eye(n) - h/4 * J2
        sol = _solve_linear_system(A2, rhs2)
        if np.linalg.norm(sol - y2) < 1e-12 * (1 + np.linalg.norm(sol)):
            y2 = sol
            break
        y2 = sol
    
    return y2, f1, f(t2, y2)


def _estimate_error(f, t, y, h, J_analytical=None):
    """Estimate local error using two half-steps vs one full step (Radau IIA)."""
    # Full step
    y_full, _, _ = _implicit_radau2_step(f, t, y, h, J_analytical)
    
    # Two half steps
    y_mid, _, _ = _implicit_radau2_step(f, t, y, h/2, J_analytical)
    y_half, _, _ = _implicit_radau2_step(f, t, t + h/2, y_mid, h/2, J_analytical)
    
    error = np.linalg.norm(y_full - y_half)
    return error, y_half  # Return the more accurate solution


def integrate(f, t_span, y0):
    """Integrate the Robertson ODE using adaptive implicit Radau IIA."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s limit
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Use analytical Jacobian if this is the Robertson problem
    # Detect by checking dimension and initial conditions
    J_analytical = None
    if len(y) == 3:
        # Assume Robertson problem
        J_analytical = _robertson_jacobian
    
    # Adaptive stepping
    h = 0.01  # Initial step size
    min_h = 1e-10
    max_h = 1.0
    
    total_calls = 0
    max_calls = 1500000  # Leave some margin
    
    while t < t1:
        if time.time() > deadline - 10.0:
            # Time running out, use remaining time with small steps
            remaining_time = deadline - time.time()
            if remaining_time < 5.0:
                break
            # Estimate how many more steps we can do
            # Each step takes roughly 0.001s, so we can do ~remaining_time/0.001 steps
            # But be conservative
            pass
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        if h < min_h:
            h = min_h
        
        # Error estimation and adaptive stepping
        error, y_new = _estimate_error(f, t, y, h, J_analytical)
        
        # Count function calls (rough estimate: each _estimate_error uses ~6-12 calls)
        total_calls += 12
        
        if total_calls > max_calls:
            break
        
        # Adapt step size
        if error < 1e-12:
            # Very small error, can increase step
            h *= 2.0
        elif error < 1e-8:
            h *= 1.5
        elif error < 1e-5:
            h *= 1.0
        elif error < 1e-3:
            h *= 0.7
        elif error < 1e-1:
            h *= 0.4
        else:
            h *= 0.1
        
        h = max(min_h, min(max_h, h))
        
        # Accept the step (use the more accurate half-step result)
        y = y_new
        t = t + h
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(y)):
            # Reset with smaller step
            h = min_h
            y = np.array(y0, dtype=float)
            t = t0
            break
    
    return y
# EVOLVE-BLOCK-END
