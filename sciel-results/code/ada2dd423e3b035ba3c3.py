"""Optimized ODE solver for 10x10 2D heat equation using matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm, eigh, eig

_START_TIME = time.time()
_TIMEOUT = 280.0  # Safety margin before 330s limit


def _reconstruct_A(f, t0, n):
    """
    Reconstruct the linear operator A for y' = A*y by probing f with basis vectors.
    Returns (A, b) where y' = A*y + b.
    """
    # Check for affine term
    y_zero = np.zeros(n)
    f_zero = f(t0, y_zero)
    b = f_zero.copy()
    
    # Probe with basis vectors to get columns of A
    A = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        f_e = f(t0, e)
        A[:, j] = f_e - b
    
    return A, b


def _solve_with_expm(f, t_span, y0):
    """
    Solve y' = A*y + b exactly using matrix exponential.
    Uses eigendecomposition for symmetric matrices for improved accuracy.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Reconstruct A and b
    A, b = _reconstruct_A(f, t0, n)
    
    try:
        # Check if A is (nearly) symmetric - common for Laplacian operators
        sym_residual = np.max(np.abs(A - A.T))
        is_symmetric = sym_residual < 1e-10 * max(1.0, np.max(np.abs(A)))
        
        if is_symmetric:
            # Use eigh for symmetric matrices - more numerically stable
            eigvals, eigvecs = eigh(A)
            # expm(A*dt) = V @ diag(exp(eigvals * dt)) @ V^{-1} = V @ diag(exp(eigvals*dt)) @ V.T
            exp_eigvals = np.exp(eigvals * dt)
            E = (eigvecs * exp_eigvals) @ eigvecs.T  # V @ diag(exp) @ V.T
        else:
            # Use general eigendecomposition or expm
            E = expm(A * dt)
        
        # Check if b is effectively zero (homogeneous system)
        b_norm = np.linalg.norm(b)
        A_norm = np.linalg.norm(A)
        if b_norm < 1e-14 * max(1.0, A_norm):
            # Homogeneous: y(t1) = E @ y0
            y_result = E @ y0
        else:
            # Non-homogeneous: use augmented matrix approach
            # But we can also use the integral formula
            # ∫₀^dt expm(A*s) b ds
            # For symmetric A with eigendecomp: expm(A*s) = V @ diag(exp(eigvals*s)) @ V.T
            # integral = V @ diag((exp(eigvals*dt) - 1) / eigvals) @ V.T @ b  (for eigvals != 0)
            
            if is_symmetric:
                # Compute integral using eigendecomposition
                # ∫₀^dt exp(λ_i * s) ds = (exp(λ_i*dt) - 1) / λ_i for λ_i != 0
                # = dt for λ_i = 0
                eigvals_nonzero = np.abs(eigvals) > 1e-14
                integral_coeff = np.zeros_like(eigvals)
                integral_coeff[eigvals_nonzero] = (np.exp(eigvals[eigvals_nonzero] * dt) - 1.0) / eigvals[eigvals_nonzero]
                integral_coeff[~eigvals_nonzero] = dt
                # integral_matrix = V @ diag(integral_coeff) @ V.T
                integral_matrix = (eigvecs * integral_coeff) @ eigvecs.T
                y_result = E @ y0 + integral_matrix @ b
            else:
                # Augmented system: [y; 1]' = [[A, b], [0, 0]] [y; 1]
                n_aug = n + 1
                M = np.zeros((n_aug, n_aug))
                M[:n, :n] = A
                M[:n, n] = b
                # Last row is all zeros
                
                E_aug = expm(M * dt)
                y_aug0 = np.zeros(n_aug)
                y_aug0[:n] = y0
                y_aug0[n] = 1.0
                
                y_final = E_aug @ y_aug0
                y_result = y_final[:n]
        
        if not np.all(np.isfinite(y_result)):
            return None
        
        return y_result
    
    except Exception:
        # Fallback: try plain expm with augmented matrix
        try:
            n_aug = n + 1
            M = np.zeros((n_aug, n_aug))
            M[:n, :n] = A
            M[:n, n] = b
            E_aug = expm(M * dt)
            y_aug0 = np.zeros(n_aug)
            y_aug0[:n] = y0
            y_aug0[n] = 1.0
            y_final = E_aug @ y_aug0
            y_result = y_final[:n]
            if np.all(np.isfinite(y_result)):
                return y_result
        except Exception:
            pass
        return None


def _rk45_adaptive(f, t0, t1, y0, rtol=1e-10, atol=1e-12, max_steps=100000):
    """
    Adaptive RK45 (Dormand-Prince) with step size control.
    """
    y = np.array(y0, dtype=float)
    t = t0
    h = (t1 - t0) / 100  # Initial step
    n = len(y)
    steps = 0
    
    # Dormand-Prince coefficients
    a2, a3, a4, a5, a6, a7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    b2 = 1/5
    b3 = 3/40, 9/40
    b4 = 44/45, -56/15, 32/9
    b5 = 19372/6561, -25360/2187, 64448/6561, -212/729
    b6 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    
    c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    
    while t < t1 - 1e-15:
        if steps > max_steps:
            break
        
        # Limit step to not exceed t1
        h = min(h, t1 - t)
        
        # RK45 stages
        k1 = f(t, y)
        k2 = f(t + a2*h, y + h*b2*k1)
        k3 = f(t + a3*h, y + h*(b3[0]*k1 + b3[1]*k2))
        k4 = f(t + a4*h, y + h*(b4[0]*k1 + b4[1]*k2 + b4[2]*k3))
        k5 = f(t + a5*h, y + h*(b5[0]*k1 + b5[1]*k2 + b5[2]*k3 + b5[3]*k4))
        k6 = f(t + a6*h, y + h*(b6[0]*k1 + b6[1]*k2 + b6[2]*k3 + b6[3]*k4 + b6[4]*k5))
        
        # 4th order solution
        y4 = y + h*(35/384*k1 + 0*k2 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
        
        # Error estimate (difference between 5th and 4th order)
        err = h*((1/384 - 35/384)*k1 + 0*k2 + (500/1113 - 500/1113)*k3 + 
                 (125/192 - 125/192)*k4 + (-2187/6784 + 2187/6784)*k5 + 
                 (11/84 - 11/84)*k6)
        # Actually, the error is:
        # err = h*(1/384*k1 - 125/192*k4 + 2187/6784*k5 - 11/84*k6)
        err = h*(1/384*k1 - 125/192*k4 + 2187/6784*k5 - 11/84*k6)
        
        err_norm = np.linalg.norm(err, ord=np.inf)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y4))
        err_ratio = err_norm / np.max(scale) if np.max(scale) > 0 else 0
        
        if err_ratio <= 1.0:
            # Accept step
            y = y4
            t += h
            steps += 1
            
            # Adjust step size
            if err_ratio > 0:
                h_new = h * min(5.0, max(0.1, 0.9 * err_ratio**(-0.2)))
            else:
                h_new = h * 5.0
            h = min(h_new, (t1 - t) if t1 > t else h)
        else:
            # Reject step, reduce h
            h = h * max(0.1, 0.9 * err_ratio**(-0.25))
            h = max(h, 1e-14)
    
    return y


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    Uses matrix exponential for linear systems, adaptive RK45 otherwise.
    """
    global _START_TIME
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try the exact matrix exponential approach first (fastest and most accurate for linear systems)
    try:
        elapsed = time.time() - _START_TIME
        if elapsed > _TIMEOUT:
            # Timeout, return best effort
            return _rk45_adaptive(f, t0, t1, y0)
        
        y_result = _solve_with_expm(f, t_span, y0)
        if y_result is not None and np.all(np.isfinite(y_result)):
            return y_result
    except Exception:
        pass
    
    # Fallback to adaptive RK45
    return _rk45_adaptive(f, t0, t1, y0)
# EVOLVE-BLOCK-END
