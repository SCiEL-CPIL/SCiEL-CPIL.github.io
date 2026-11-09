"""Improved program: Analytical solution via generalized eigenvalue decomposition with L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 200.0  # Safety margin before 210s kill

    # Hermite polynomial values at 0 (physicists' convention)
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 1039500.0

    # Variance (effective n) for each Hermite mode: <x^2> = n + 1/2
    n0 = 0.5
    n4 = 4.5
    n8 = 8.5
    n12 = 12.5

    # Coefficients for the constraint c12 = -k * (c0*h0_0 + c4*h4_0 + c8*h8_0)
    k = 1.0 / h12_0
    a = np.array([h0_0, h4_0, h8_0])

    # Matrix A for the numerator of the variance
    D = np.diag([n0, n4, n8])
    A = D + n12 * (k**2) * np.outer(a, a)

    # Matrix B for the denominator of the variance
    I = np.eye(3)
    B = I + (k**2) * np.outer(a, a)

    # Solve the generalized eigenvalue problem A v = lambda B v
    best_val = np.inf
    best_vec = None

    try:
        eigvals, eigvecs = np.linalg.eigh(A, B)
        idx_min = np.argmin(eigvals)
        best_val = eigvals[idx_min]
        best_vec = eigvecs[:, idx_min].copy()
    except Exception:
        pass

    # If we have time, do a high-precision polish with L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0 and best_vec is not None:
        try:
            from scipy.optimize import minimize
            
            def rayleigh_quotient(c):
                c = np.asarray(c, dtype=float)
                num = c @ A @ c
                den = c @ B @ c
                if den <= 0:
                    return 1e10
                return num / den

            def rayleigh_grad(c):
                c = np.asarray(c, dtype=float)
                Ac = A @ c
                Bc = B @ c
                num = c @ Ac
                den = c @ Bc
                if den <= 0:
                    return np.zeros_like(c)
                # Gradient of (c^T A c) / (c^T B c)
                # d/dc (c^T A c) = 2 A c
                # d/dc (c^T B c) = 2 B c
                # Quotient rule: (2 Ac * den - 2 Bc * num) / den^2
                grad = (2 * (Ac * den - Bc * num)) / (den * den)
                return grad

            # Try a few different starting points
            starts = [
                best_vec,
                -best_vec,
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([1.0, 1.0, 1.0]),
                np.array([-1.0, 1.0, -1.0]),
                np.array([1.0, -1.0, 1.0]),
            ]
            
            for s in starts:
                if time.time() - start_time > DEADLINE - 3.0:
                    break
                res = minimize(
                    rayleigh_quotient,
                    s,
                    jac=rayleigh_grad,
                    method='L-BFGS-B',
                    options={'gtol': 1e-12, 'maxiter': 1000}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_vec = res.x.copy()
        except Exception:
            pass

    if best_vec is None:
        # Fallback to seed
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    return np.array(best_vec)
# EVOLVE-BLOCK-END
