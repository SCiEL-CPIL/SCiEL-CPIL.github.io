"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Start from rank-42 Strassen block decomposition, then attempt
rank reduction via numerical optimization. If reduction fails, fall back
to the known rank-42 solution.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7).
    U1 row: i*2+j, V1 row: j*2+k, W1 row: k*2+i (trace convention)."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1  # C00 += M1
    W1[3, 0] = 1  # C11 += M1

    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1  # C10 += M2
    W1[3, 1] = -1 # C11 -= M2

    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1  # C01 += M3
    W1[3, 2] = 1  # C11 += M3

    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1  # C00 += M4
    W1[1, 3] = 1  # C10 += M4

    # M5 = (A00+A01)B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1 # C00 -= M5
    W1[2, 4] = 1  # C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1  # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1  # C00 += M7

    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 Strassen blocks."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42

    U = np.zeros((n * m, total_rank), dtype=float)  # (8, 42)
    V = np.zeros((m * p, total_rank), dtype=float)  # (24, 42)
    W = np.zeros((p * n, total_rank), dtype=float)  # (12, 42)

    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local
                    for s in range(7):
                        coeff = U1[i * 2 + j_local, s]
                        if coeff != 0:
                            U[i * m + a, col_offset + s] += coeff

            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local, s]
                        if coeff != 0:
                            V[a * p + bcol, col_offset + s] += coeff

            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i, s]
                        if coeff != 0:
                            W[bcol * n + i, col_offset + s] += coeff

    return U, V, W


def _tensor_residual(U, V, W, n, m, p):
    """Compute max |sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] - 1| over all (i,j,k)."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _try_rank_reduction(U0, V0, W0, n, m, p, target_rank, deadline, max_iters=500):
    """
    Try to find a rank-`target_rank` decomposition by optimizing from a
    random projection of the rank-42 solution. Uses gradient-free Nelder-Mead
    on the residual.
    Returns (U, V, W) if residual < 1e-9, else None.
    """
    if time.time() > deadline:
        return None
    
    r = target_rank
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    # Initialize by taking random linear combinations of the 42 rank-42 terms
    # to get r effective terms
    np.random.seed(42)
    # For each of the r output terms, pick a random combination of the 42 input terms
    for s in range(r):
        # Random weights over the 42 original terms
        w = np.random.randn(42)
        w /= (np.linalg.norm(w) + 1e-15)
        U[:, s] = U0 @ w
        V[:, s] = V0 @ w
        W[:, s] = W0 @ w
    
    # Nelder-Mead style optimization on the residual
    def residual(params):
        # params: flattened (U, V, W)
        Uf = params[:n*m*r].reshape(n*m, r)
        Vf = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        Wf = params[n*m*r + m*p*r:].reshape(p*n, r)
        return _tensor_residual(Uf, Vf, Wf, n, m, p)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    f0 = residual(x0)
    if f0 < 1e-9:
        return U, V, W
    
    # Simple iterative refinement: use scipy's minimize if available, else manual
    try:
        from scipy.optimize import minimize
        result = minimize(residual, x0, method='Nelder-Mead',
                         options={'maxiter': max_iters, 'xatol': 1e-12, 'fatol': 1e-12,
                                  'adaptive': True})
        if time.time() > deadline:
            return None
        xf = result.x
        Uf = xf[:n*m*r].reshape(n*m, r)
        Vf = xf[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        Wf = xf[n*m*r + m*p*r:].reshape(p*n, r)
        if result.fun < 1e-9:
            return Uf, Vf, Wf
    except Exception:
        pass
    
    return None


def _verify(U, V, W, n, m, p):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _to_output(mat):
    """Convert numpy array to list of lists with int where possible."""
    return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]

def _build_schoolbook(n, m, p):
    """Build schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj][col] = 1
                V[jj * p + kk][col] = 1
                W[kk * n + ii][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the known rank-42 solution
    U42, V42, W42 = _build_rank42()
    
    if not _verify(U42, V42, W42, n, m, p):
        # Fallback: schoolbook rank 48
        U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to reduce rank below 42 via numerical optimization
    best_U, best_V, best_W = U42, V42, W42
    best_rank = 42
    
    # Try target ranks from 36 up to 41
    for target_rank in range(36, 42):
        if time.time() > deadline - 15.0:
            break
        result = _try_rank_reduction(U42, V42, W42, n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    return {"U": _to_output(best_U), "V": _to_output(best_V), "W": _to_output(best_W)}
