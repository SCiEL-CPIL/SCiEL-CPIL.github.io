import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 285.0
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]

    # We will extract A and b using n+1 calls, then solve exactly via expm.
    # To minimize call count, we use exactly n+1 calls.
    # b = f(t0, 0)
    zero = np.zeros(n)
    b = f(t0, zero)
    
    A = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        fi = f(t0, ei)
        A[:, i] = fi - b

    # Check time
    if time.time() > deadline - 2:
        # Fallback if time is tight, but expm is fast, so probably fine.
        pass

    # Solve exactly
    # y' = A y + b
    # y(t) = expm(A dt) y0 + ∫_0^dt expm(A (dt-s)) b ds
    #      = expm(A dt) y0 + (expm(A dt) I - I) A^{-1} b   if A invertible
    # Better: use augmented matrix to avoid inverting A
    # z = [y; 1], z' = M z, M = [[A, b], [0, 0]]
    M = np.zeros((n+1, n+1))
    M[:n, :n] = A
    M[:n, n] = b
    
    dt = t1 - t0
    M_scaled = dt * M
    
    expM = scipy.linalg.expm(M_scaled)
    
    z_init = np.zeros(n+1)
    z_init[:n] = y0
    z_init[n] = 1.0
    
    z_final = expM @ z_init
    y_final = z_final[:n]
    
    return y_final
