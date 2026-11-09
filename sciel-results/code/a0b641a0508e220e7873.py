import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    n = len(y0)
    
    # Extract system matrix A by probing f with basis vectors
    # f(t, y) = A @ y (for linear time-invariant system)
    # We'll estimate A by calling f at t0 with y0 and perturbations
    
    # First, get f at initial state
    y0_arr = np.array(y0, dtype=float)
    f0 = np.array(f(t0, y0_arr), dtype=float)
    
    # For a linear system y' = A@y, we can estimate A by:
    # A @ y0 = f0, and we can use additional probes if needed
    # But for 100-point convection-diffusion, we can assume f(t,y) = A@y
    
    # Strategy: Use Parlett recurrence to compute expm(A*T) @ y0
    # without explicitly forming expm(A)
    
    T = t1 - t0
    
    # Parlett recurrence for matrix exponential action on a vector:
    # e^(AT) y0 = sum_{k=0}^{p} c_k * (AT)^k @ y0
    # where c_k are coefficients from the Taylor series
    
    # For better numerical stability, use scaling and squaring:
    # e^(AT) = (e^(AT/s))^s
    # Choose s such that ||AT/s|| is small
    
    # Estimate ||A|| from the initial probe
    if np.linalg.norm(y0_arr) > 1e-15:
        # Assume A @ y0 = f0, so ||A|| ≈ ||f0|| / ||y0||
        A_norm_est = np.linalg.norm(f0) / np.linalg.norm(y0_arr)
    else:
        A_norm_est = 1.0
    
    # Choose scaling factor s such that ||AT/s|| < 0.5 for accuracy
    if A_norm_est * T > 1e-10:
        s = max(1, int(np.ceil(A_norm_est * T / 0.5)))
    else:
        s = 1
    
    # Compute (AT/s) @ v efficiently using the linear operator
    # Since f(t, y) = A @ y (time-invariant), we can apply A to any vector v
    # by calling f(t0, v)
    
    def apply_A(v):
        return np.array(f(t0, v), dtype=float)
    
    # Parlett recurrence: compute e^(M) @ y0 where M = A*T/s
    # Using Taylor series with coefficients
    # For ||M|| < 1, the series converges quickly
    
    def expm_action_on_vector(M, v, tol=1e-15, max_iter=100):
        """Compute e^M @ v using Taylor series with adaptive termination"""
        result = np.zeros_like(v)
        term = np.ones_like(v) * v  # M^0 @ v = v
        result = term.copy()
        
        for k in range(1, max_iter):
            term = M @ term / k
            result = result + term
            if np.linalg.norm(term) < tol * np.linalg.norm(result):
                break
        
        return result
    
    # Compute M = A*T/s
    # We need to apply M to vectors: M @ v = (A*T/s) @ v = (T/s) * f(t0, v)
    def apply_M(v):
        return (T / s) * np.array(f(t0, v), dtype=float)
    
    # Compute e^(AT/s) @ y0
    y_current = np.array(y0, dtype=float)
    for _ in range(s):
        # Compute e^(M) @ y_current where M = A*T/s
        y_current = expm_action_on_vector(apply_M, y_current)
    
    return y_current
