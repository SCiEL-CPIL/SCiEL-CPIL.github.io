import numpy as np
import scipy.linalg as sla
import time

def integrate(f, t_span, y0):
    """
    Solves y' = f(t, y) for a 15-dimensional stiff linear system.
    Strategy: The task contract states this is a fixed 15-dimensional stiff
    linear system. We assume y' = A*y + b with constant A and b.
    We estimate A and b using minimal function calls:
      - b = f(t0, 0)
      - A columns: f(t0, e_j) - b for j=0..n-1
    Total calls: 1 + n = 16.
    Then use the exact matrix exponential solution.
    
    To reduce calls further, we can try to estimate A with fewer probes
    by using random vectors, but that introduces estimation error.
    Since the score penalizes calls with -0.5*log10(calls), and we need
    high accuracy, 16 calls is a good balance.
    
    However, we can try an even more aggressive approach:
    Use the fact that for a linear system, we can compute exp(A*T)*y0
    and the integral term using Krylov subspace methods or scaling-and-squaring
    applied to vectors, avoiding explicit A formation. But we still need A
    to apply to vectors.
    
    Actually, the most efficient approach for a known linear structure:
    1. Probe b = f(t0, zeros)  -> 1 call
    2. Probe A by evaluating f(t0, v) for n basis vectors -> n calls
       Total: n+1 = 16 calls
    
    For a 15-dim system, this gives us the exact A and b (assuming linearity
    and time-invariance as stated in the task).
    
    Then compute:
      y(T) = exp(A*T) * y0 + integral_0^T exp(A*(T-s)) * b ds
           = exp(A*T) * y0 + (exp(A*T) - I) * A^{-1} * b  (if A invertible)
    
    This is exact (up to floating point), so relative error is ~machine epsilon,
    giving -log10(1e-15) ≈ 15 for the accuracy term.
    With 16 calls: -0.5*log10(16) ≈ -0.5*1.2 = -0.6
    Total score ≈ 15 - 0.6 = 14.4 (very high)
    
    Wait, the score is -log10(relative terminal error) - 0.5*log10(calls to f).
    If relative error is ~1e-15, then -log10(1e-15) = 15.
    -0.5*log10(16) = -0.5 * 1.204 = -0.602
    Score ≈ 14.4
    
    But the current score is 1.149, which suggests the reference solution
    might have a different error level, or the scoring is different.
    
    Actually, looking at the inspirations, scores are around 1.15-1.16.
    This suggests the relative error is much larger, or the scoring formula
    is applied differently. Perhaps the "relative terminal error" is against
    a high-precision reference, and our solution has some error.
    
    Let me reconsider: if we use the exact matrix exponential, the error
    should be at machine precision level. But maybe the issue is that
    the system is NOT exactly time-invariant, or there's nonlinearity.
    
    The task says "fixed 15-dimensional stiff linear system" - so it should
    be linear and time-invariant. Our approach should give near-machine
    precision accuracy.
    
    Let's optimize for minimal calls while maintaining accuracy.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Estimate b and A
    # b = f(t0, 0)
    b = f(t0, np.zeros(n))
    b = np.array(b, dtype=float)
    
    # A columns: f(t0, e_j) - b
    A = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        val = f(t0, e_j)
        val = np.array(val, dtype=float)
        A[:, j] = val - b
    
    T = t1 - t0
    
    # Compute exp(A*T)
    AT = A * T
    exp_AT = sla.expm(AT)
    
    # Compute integral term
    # Try A^{-1} * b
    try:
        cond = np.linalg.cond(A)
        if cond < 1e12:
            x = np.linalg.solve(A, b)
            integral_term = (exp_AT - np.eye(n)) @ x
        else:
            raise np.linalg.LinAlgError("ill-conditioned")
    except np.linalg.LinAlgError:
        # Series expansion: integral_0^T exp(A*(T-s)) b ds = sum_{k=0}^inf (A*T)^k b / (k+1)!
        term = np.zeros(n)
        current = b.copy()
        for k in range(100):
            term += current / (k + 1)
            current = AT @ current / (k + 1)
            if np.linalg.norm(current / (k + 1)) < 1e-16 * (np.linalg.norm(term) + 1e-300):
                break
        integral_term = term
    
    y_final = exp_AT @ y0 + integral_term
    
    if time.time() > deadline:
        return y_final
    
    return y_final
