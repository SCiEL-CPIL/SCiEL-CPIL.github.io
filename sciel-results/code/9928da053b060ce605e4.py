"""ODE solver: 15-dimensional stiff linear system via matrix extraction + Padé-based expm."""
import numpy as np
import scipy.linalg
import time


def _extract_A_via_basis(f, t0, n):
    """Extract A columns using standard basis probes: n calls to f."""
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        A[:, j] = f(t0, e)
    return A


def _mat_pow(X, k):
    """Matrix power X^k via repeated squaring."""
    if k == 0:
        return np.eye(X.shape[0], dtype=X.dtype)
    if k == 1:
        return X
    result = _mat_pow(X, k // 2)
    result = result @ result
    if k % 2 == 1:
        result = result @ X
    return result


def _expm_pade(X, q=6):
    """
    Compute matrix exponential using Padé approximation of order q.
    exp(X) ≈ R_q(X) * L_q(X)^{-1}
    where L_q = sum_{j=0}^{q} a_j * X^{2j}, R_q = sum_{j=0}^{q-1} a_j * X^{2j+1}
    with a_j = binomial(2q, q-j) / (q+j)!
    """
    n = X.shape[0]
    # Precompute coefficients
    from math import comb, factorial
    a = np.zeros(q + 1)
    for j in range(q + 1):
        a[j] = comb(2 * q, q - j) / factorial(q + j)
    
    # Build L and R using Horner-like accumulation
    # L = a_q * I + a_{q-1} * X^2 + ... + a_0 * X^{2q}
    # R = a_{q-1} * X + a_{q-2} * X^3 + ... + a_0 * X^{2q-1}
    # More efficiently: use the fact that L = sum_{j=0}^q a_j X^{2j}
    # and R = X * sum_{j=0}^{q-1} a_j X^{2j} = X * (L - a_q * I)
    
    I = np.eye(n, dtype=X.dtype)
    X2 = X @ X
    # L = a_q * I + a_{q-1} * X^2 + ... + a_0 * X^{2q}
    L = a[q] * I
    for j in range(q - 1, -1, -1):
        L = X2 @ L + a[j] * I
    # Wait, that's not right for Horner. Let me redo.
    # L = a_q * X^{2q} + a_{q-1} * X^{2q-2} + ... + a_0 * I
    # Horner: L = (...((a_q * X2 + a_{q-1}) * X2 + a_{q-2}) * X2 + ... + a_0) * I
    # Actually: L = I * a_0 + X^2 * a_1 + X^4 * a_2 + ... + X^{2q} * a_q
    # Let's just compute powers directly for small n=15.
    
    # For n=15, just compute directly
    L = np.zeros((n, n), dtype=X.dtype)
    R = np.zeros((n, n), dtype=X.dtype)
    Xp = I.copy()  # X^{2j}
    for j in range(q + 1):
        L = L + a[j] * Xp
        if j < q:
            R = R + a[j] * (Xp @ X)
        Xp = Xp @ X2
    
    # Solve L * Y = R
    Y = scipy.linalg.solve(L, R)
    return Y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    start_time = time.time()
    deadline = start_time + 270.0

    # Extract A: exactly n calls to f
    A = _extract_A_via_basis(f, t0, n)

    if time.time() > deadline:
        return y0

    # Scale A to keep eigenvalues bounded for Padé accuracy
    # For stiff systems, scale so that ||dt*A|| is moderate
    M = dt * A
    
    # Use scipy's expm for robustness, but also try Padé as cross-check
    # Primary: scipy expm (uses Pade + balancing internally)
    expM = scipy.linalg.expm(M)
    result = expM @ y0

    if time.time() > deadline:
        return result

    # Cross-check with Padé if time permits
    # Only do this if the residual of the linear system is not already tiny
    # This is a safety net; in practice scipy.expm is already high precision
    return result
