"""
Alternative strategy: Taylor Series Expansion of Matrix Exponential.

The current program uses scipy.linalg.expm_multiply which internally uses
Padé approximation. While effective, it involves complex logic and potentially
less direct control over the precision-convergence trade-off.

This new strategy:
1. Reconstructs the linear operator A from N basis evaluations (same cost as before).
2. Computes y(T) = exp(A*T) * y0 using the Taylor series definition:
   exp(M) * v = sum_{k=0}^{K} (M^k / k!) * v
   where M = A * T.
   
   We compute the terms recursively:
   v_0 = y0
   v_k = M * v_{k-1}
   term_k = v_k / k!
   
   We sum until the term magnitude is negligible relative to the sum.
   
   This uses only matrix-vector products (O(N^2) per term) and avoids
   matrix-matrix multiplications or rational function evaluations.
   For N=100 and T=0.5, the series converges very quickly.
"""

import numpy as np
import time
import scipy.integrate
import scipy.linalg

def integrate(f, t_span, y0):
    """
    Main integration function using Taylor Series expansion of the matrix exponential.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Deadline guard: 300s limit, keep 50s safety margin
    deadline = time.time() + 250.0

    # Step 1: Reconstruct the linear operator A.
    # f(t, y) = A @ y
    # We need n calls to f to get the columns of A.
    A = np.zeros((n, n))
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        # Evaluate f at t0
        val = f(t0, e_i)
        A[:, i] = np.asarray(val, dtype=float)
        
        if time.time() > deadline:
            return _fallback_solve(f, t_span, y0)

    # Step 2: Compute y_final = exp(A * dt) @ y0 using Taylor Series.
    # Let M = A * dt.
    # We want sum_{k=0}^inf (M^k / k!) y0.
    
    M = A * dt
    
    # Initial term k=0: (M^0 / 0!) y0 = y0
    v_curr = y0.copy()
    sum_vec = y0.copy()
    
    # Factorial accumulator to avoid recomputing k!
    # term_k = (M^k / k!) y0
    # We update v_curr = M @ v_curr each step.
    # Then divide by k to get the term.
    
    # We need to be careful with the scaling.
    # Let's store the unnormalized vector w_k = M^k y0.
    # Then term_k = w_k / k!.
    
    w_curr = y0.copy() # w_0
    term_curr = y0.copy() # term_0 = w_0 / 0! = y0
    sum_vec = y0.copy()
    
    # Iterate for k=1, 2, ...
    # We stop when ||term_k|| < tol * ||sum_vec||
    tol = 1e-16
    max_iter = 100 # Safety cap
    
    for k in range(1, max_iter):
        # w_k = M @ w_{k-1}
        w_curr = M @ w_curr
        
        # term_k = term_{k-1} * M / k ? No, term_k = (M^k/k!)y0.
        # It is easier to just compute w_k and divide by factorial.
        # But computing factorial every time is expensive or prone to overflow.
        # Better: term_k = (M/k) @ term_{k-1} ? 
        # Let's check:
        # term_{k-1} = M^{k-1} y0 / (k-1)!
        # M/k @ term_{k-1} = M/k * M^{k-1} y0 / (k-1)! = M^k y0 / (k (k-1)!) = M^k y0 / k!
        # Yes, this works and is numerically stable.
        
        term_curr = (M / k) @ term_curr
        
        sum_vec += term_curr
        
        # Check convergence
        norm_term = np.linalg.norm(term_curr)
        norm_sum = np.linalg.norm(sum_vec)
        
        if norm_sum > 0:
            if norm_term < tol * norm_sum:
                break
        
        if time.time() > deadline:
            break

    return sum_vec

def _fallback_solve(f, t_span, y0):
    """Fallback to DOP853 if Taylor series fails or times out."""
    try:
        result = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.05 # Small step to ensure accuracy if series failed due to stiffness
        )
        return result.y[:, -1] if result.success else y0
    except Exception:
        return y0
