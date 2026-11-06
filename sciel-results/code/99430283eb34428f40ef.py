"""ODE solver: 15-dimensional stiff linear system using Taylor series matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on [t0, t1] for a 15-dim stiff linear system.
    
    Strategy: Extract the constant linear operator A and offset b from f,
    then compute y(t1) = expm((t1-t0)*A) @ y0 + integral of expm((t1-t)*A) @ b dt
    using high-order Taylor series for the matrix exponential.
    """
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Extract A and b: assume f(t, y) = A @ y + b (constant coefficients)
    # Evaluate at y=0 to get b
    b = np.asarray(f(t0, np.zeros(n)), dtype=float)
    
    # Evaluate at basis vectors to extract A
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        Aj = np.asarray(f(t0, e_j), dtype=float)
        A[:, j] = Aj - b  # f(t, e_j) = A @ e_j + b = A[:, j] + b
    
    # Compute M = dt * A
    M = dt * A
    
    # Determine number of Taylor terms needed based on ||M||
    norm_M = np.linalg.norm(M, 1)
    K = max(50, int(norm_M * 1.5) + 50)
    K = min(K, 200)
    
    # Compute expm(M) via Taylor series: expm(M) = sum_{k=0}^{K} M^k / k!
    I = np.eye(n)
    expM = np.zeros((n, n), dtype=float)
    term = I.copy()
    for k in range(K + 1):
        expM += term
        if k < K:
            term = term @ M / (k + 1)
        if k > 10 and np.linalg.norm(term, 1) < 1e-15 * np.linalg.norm(expM, 1):
            break
    
    # Compute integral part: sum_{k=0}^{K} (dt^{k+1} / (k+1)!) * A^k @ b
    integral_part = np.zeros(n, dtype=float)
    Ak_b = b.copy()  # A^0 @ b
    coeff = dt  # coefficient for k=0: dt^1 / 1!
    
    for k in range(K + 1):
        integral_part += coeff * Ak_b
        if k < K:
            Ak_b = A @ Ak_b
            coeff = coeff * dt / (k + 2)  # next coefficient: dt^{k+2} / (k+2)!
    
    y_final = expM @ y0 + integral_part
    
    return y_final
# EVOLVE-BLOCK-END
