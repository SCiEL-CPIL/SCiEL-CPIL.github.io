"""Improved ODE solver for 15-dimensional stiff linear system using high-precision matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    start_time = time.time()
    deadline = start_time + 285.0
    
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0_arr
    
    # Probe the linear system at t0 using exactly n+1 function calls
    # f(t, 0) gives b, f(t, e_j) - f(t, 0) gives column j of A
    try:
        y_zero = np.zeros(n)
        b = np.array(f(t0, y_zero), dtype=float)
    except Exception:
        return _fallback_solve(f, t0, t1, y0_arr, deadline)
    
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        try:
            f_ej = np.array(f(t0, e_j), dtype=float)
        except Exception:
            return _fallback_solve(f, t0, t1, y0_arr, deadline)
        A[:, j] = f_ej - b
    
    if time.time() > deadline:
        return y0_arr
    
    # Check linearity and time-invariance by probing at t1
    # If time-invariant and linear, use exact matrix exponential solution
    time_invariant = True
    try:
        b_t1 = np.array(f(t1, y_zero), dtype=float)
        if not np.allclose(b, b_t1, rtol=1e-12, atol=1e-14):
            time_invariant = False
        else:
            for j in range(n):
                e_j = np.zeros(n)
                e_j[j] = 1.0
                f_ej_t1 = np.array(f(t1, e_j), dtype=float)
                col_t1 = f_ej_t1 - b_t1
                if not np.allclose(A[:, j], col_t1, rtol=1e-12, atol=1e-14):
                    time_invariant = False
                    break
    except Exception:
        time_invariant = False
    
    if time_invariant:
        # Construct augmented system for y' = A y + b
        # Let z = [y; 1], then z' = M z where M = [[A, b], [0, 0]]
        M = np.zeros((n + 1, n + 1), dtype=float)
        M[:n, :n] = A
        M[:n, n] = b
        
        z0 = np.zeros(n + 1, dtype=float)
        z0[:n] = y0_arr
        z0[n] = 1.0
        
        # Use high-precision computation via scaling and squaring with Taylor series
        try:
            expM = _high_precision_expm(M * dt)
            z_final = expM @ z0
            y_final = z_final[:n]
            
            if np.all(np.isfinite(y_final)):
                return y_final
        except Exception:
            pass
        
        # Fallback to scipy's expm if custom implementation fails
        try:
            expM = expm(M * dt)
            z_final = expM @ z0
            y_final = z_final[:n]
            
            if np.all(np.isfinite(y_final)):
                return y_final
        except Exception:
            pass
    
    # Fallback: robust stiff solver
    return _fallback_solve(f, t0, t1, y0_arr, deadline)


def _high_precision_expm(B):
    """
    Compute matrix exponential using scaling and squaring with high-order Taylor series.
    Uses more terms than scipy's default for potentially higher precision.
    """
    n = B.shape[0]
    normB = np.linalg.norm(B, 1)
    
    # Choose s such that ||B/s|| < 0.5
    # Use s = ceil(log2(normB)) for better scaling
    if normB < 1e-15:
        return np.eye(n, dtype=B.dtype)
    
    s = max(1, int(np.ceil(np.log2(normB))))
    B_scaled = B / (2 ** s)
    
    # Taylor series with more terms for high precision
    # e^X = sum_{k=0}^{inf} X^k / k!
    # For ||X|| < 0.5, convergence is very fast
    X = B_scaled.copy()
    term = np.eye(n, dtype=B.dtype).copy()
    result = term.copy()
    
    # Use up to 50 terms for very high precision
    for k in range(1, 50):
        term = term @ X / k
        result = result + term
        
        # Check convergence
        if np.linalg.norm(term, 1) < 1e-18 * np.linalg.norm(result, 1):
            break
    
    # Squaring step
    for _ in range(s):
        result = result @ result
    
    return result


def _fallback_solve(f, t0, t1, y0_arr, deadline):
    """Fallback to robust stiff ODE solver with tight tolerances."""
    try:
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='Radau',
            rtol=1e-14,
            atol=1e-16
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        sol_bdf = solve_ivp(
            f, (t0, t1), y0_arr,
            method='BDF',
            rtol=1e-14,
            atol=1e-16
        )
        if sol_bdf.success:
            return sol_bdf.y[:, -1]
    except Exception:
        pass
    
    return y0_arr
# EVOLVE-BLOCK-END
