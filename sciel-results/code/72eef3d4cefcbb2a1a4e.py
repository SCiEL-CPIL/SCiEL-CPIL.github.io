import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _check_decomposition(U, V, W, T, tol=1e-8):
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    R = U_arr.shape[1]
    # Vectorized: compute all entries at once
    # result[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    result = np.einsum('ur,vr,wr->uvw', U_arr, V_arr, W_arr)
    # Only check the 20 nonzero positions
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(result[u, v, w] - T[u, v, w]) > tol:
                    return False
    # Check that zero positions are also close to zero (important for validity)
    # The tensor only has 20 nonzeros out of 4*10*10=400 positions
    # For a valid decomposition, the other 380 should be ~0
    mask = np.ones((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                mask[u, v, w] = False
    if np.max(np.abs(result[mask])) > tol:
        return False
    return True


def _check_exact(U, V, W, T):
    def to_frac(x):
        hx = round(x * 2)
        if abs(x - hx / 2.0) < 1e-7:
            return Fraction(int(hx), 2)
        f = Fraction(x).limit_denominator(128)
        if abs(x - float(f)) < 1e-7:
            return f
        return Fraction(x).limit_denominator(128)

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = len(U[0])
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _snap_to_rational(U, V, W, T):
    """Try to snap to small-denominator rationals and verify exactly."""
    for denom in [1, 2, 4]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
    return None


def _build_strassen_block(col0, col1, base, U, V, W, n, m, p):
    r = base
    U[0][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 1
    U[2][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1
    W[col0*n+1][r] += 1; W[col1*n+1][r] += -1

    r = base + 2
    U[0][r] += 1
    V[0*p+col1][r] += 1; V[1*p+col1][r] += -1
    W[col1*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 3
    U[3][r] += 1
    V[1*p+col0][r] += 1; V[0*p+col0][r] += -1
    W[col0*n+0][r] += 1; W[col0*n+1][r] += 1

    r = base + 4
    U[0][r] += 1; U[1][r] += 1
    V[1*p+col1][r] += 1
    W[col0*n+0][r] += -1; W[col1*n+0][r] += 1

    r = base + 5
    U[2][r] += 1; U[0][r] += -1
    V[0*p+col0][r] += 1; V[0*p+col1][r] += 1
    W[col1*n+1][r] += 1

    r = base + 6
    U[1][r] += 1; U[3][r] += -1
    V[1*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1


def _build_rank18():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    _build_strassen_block(0, 1, 0, U, V, W, n, m, p)
    _build_strassen_block(2, 3, 7, U, V, W, n, m, p)

    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1

    return U, V, W


def _als_step(U, V, W, T, deadline, sparsity_lambda=0.0):
    """One ALS iteration with optional soft-L1 sparsity penalty."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    R = U.shape[1]

    # Update U
    A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
    for u in range(U_rows):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        if sparsity_lambda > 0:
            # Soft-thresholding approximation
            for r in range(R):
                if abs(sol[r]) < sparsity_lambda:
                    sol[r] = 0.0
        U[u] = sol

    # Update V
    A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
    for v in range(V_rows):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        if sparsity_lambda > 0:
            for r in range(R):
                if abs(sol[r]) < sparsity_lambda:
                    sol[r] = 0.0
        V[v] = sol

    # Update W
    A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
    for w in range(W_rows):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        if sparsity_lambda > 0:
            for r in range(R):
                if abs(sol[r]) < sparsity_lambda:
                    sol[r] = 0.0
        W[w] = sol

    return U, V, W


def _try_als_from_init(U0, V0, W0, T, deadline, max_iters=60, sparsity_lambda=0.0):
    """Run ALS from a given initialization."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    R = U0.shape[1]
    U, V, W = U0.copy(), V0.copy(), W0.copy()
    prev_err = np.inf

    for it in range(max_iters):
        if time.time() > deadline:
            break
        U, V, W = _als_step(U, V, W, T, deadline, sparsity_lambda)
        result = np.einsum('ur,vr,wr->uvw', U, V, W)
        err = np.linalg.norm(result - T.astype(np.float64))
        if abs(prev_err - err) < 1e-14:
            break
        prev_err = err

    if _check_decomposition(U, V, W, T, tol=1e-7):
        snapped = _snap_to_rational(U, V, W, T)
        if snapped is not None:
            return snapped
        return U.tolist(), V.tolist(), W.tolist()
    return None


def _try_rank_revealing_als(T, R, deadline, n, m, p):
    """
    Rank-revealing approach: start from rank-18 solution (zero-padded to R),
    then use sparsity-inducing ALS to prune terms.
    """
    if R >= 18:
        return None  # We already have rank 18

    U18, V18, W18 = _build_rank18()
    U18_arr = np.array(U18, dtype=np.float64)
    V18_arr = np.array(V18, dtype=np.float64)
    W18_arr = np.array(W18, dtype=np.float64)

    # Zero-pad to rank R
    U0 = np.zeros((n * m, R), dtype=np.float64)
    V0 = np.zeros((m * p, R), dtype=np.float64)
    W0 = np.zeros((p * n, R), dtype=np.float64)
    U0[:, :18] = U18_arr
    V0[:, :18] = V18_arr
    W0[:, :18] = W18_arr

    # Try with decreasing sparsity penalties
    for lam in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
        if time.time() > deadline:
            break
        result = _try_als_from_init(U0, V0, W0, T, deadline, max_iters=80,
                                     sparsity_lambda=lam)
        if result is not None:
            return result

    # Try random perturbations of the rank-18 init
    for seed in [42, 123, 7, 999, 2024, 555, 888, 111]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        U0 = U18_arr.copy()
        V0 = V18_arr.copy()
        W0 = W18_arr.copy()
        # Add small noise to break symmetry
        U0 += np.random.randn(n * m, 18) * 0.01
        V0 += np.random.randn(m * p, 18) * 0.01
        W0 += np.random.randn(p * n, 18) * 0.01

        # Zero-pad
        U0p = np.zeros((n * m, R), dtype=np.float64)
        V0p = np.zeros((m * p, R), dtype=np.float64)
        W0p = np.zeros((p * n, R), dtype=np.float64)
        U0p[:, :18] = U0
        V0p[:, :18] = V0
        W0p[:, :18] = W0

        result = _try_als_from_init(U0p, V0p, W0p, T, deadline, max_iters=60,
                                     sparsity_lambda=0.0)
        if result is not None:
            return result

    return None


def _try_random_als(T, R, deadline, n, m, p, n_seeds=10):
    """Try ALS with random initializations for a given rank."""
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    seeds = [42, 123, 7, 999, 2024, 555, 888, 111, 333, 777,
             1234, 4321, 5678, 8765, 2468, 1357, 9012, 3456, 7890, 12345]

    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        for scale in [0.1, 0.5, 1.0, 2.0]:
            if time.time() > deadline:
                break
            U0 = np.random.randn(U_rows, R) * scale
            V0 = np.random.randn(V_rows, R) * scale
            W0 = np.random.randn(W_rows, R) * scale
            result = _try_als_from_init(U0, V0, W0, T, deadline, max_iters=50)
            if result is not None:
                return result
    return None


def _try_structured_rank17(T, deadline, n, m, p):
    """
    Structured search for rank 17: try to merge the 4 schoolbook column-4 terms
    into the Strassen blocks by modifying the block equations.
    
    The key idea: the 4 schoolbook terms for column 4 are:
      A[0,0]*B[0,4] -> C[0,4]
      A[0,1]*B[1,4] -> C[0,4]
      A[1,0]*B[0,4] -> C[1,4]
      A[1,1]*B[1,4] -> C[1,4]
    
    We try to absorb these into the existing Strassen terms by adding
    column-4 contributions to the V and W factors.
    """
    # Start from rank-18 and try to eliminate one term at a time
    U18, V18, W18 = _build_rank18()
    U18_arr = np.array(U18, dtype=np.float64)
    V18_arr = np.array(V18, dtype=np.float64)
    W18_arr = np.array(W18, dtype=np.float64)

    # For each of the 18 terms, try to zero it out by adjusting others
    for term_to_remove in range(18):
        if time.time() > deadline:
            break
        # Create rank-17 by removing this term
        U17 = U18_arr[:, [r for r in range(18) if r != term_to_remove]]
        V17 = V18_arr[:, [r for r in range(18) if r != term_to_remove]]
        W17 = W18_arr[:, [r for r in range(18) if r != term_to_remove]]

        # Try to fix the decomposition by small adjustments
        # Use ALS starting from this initialization
        result = _try_als_from_init(U17, V17, W17, T, deadline, max_iters=80,
                                     sparsity_lambda=0.0)
        if result is not None:
            return result

    return None


def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    t0 = time.time()
    deadline = t0 + 138  # 12s safety margin before 150s kill

    n, m, p = DIMENSIONS
    T = _build_target()

    # Build and verify rank-18 (guaranteed baseline)
    U18, V18, W18 = _build_rank18()
    if not _check_decomposition(U18, V18, W18, T, tol=1e-8):
        # Should never happen, but fallback to schoolbook
        U_l, V_l, W_l = _schoolbook_decomposition()
        return {"U": U_l, "V": V_l, "W": W_l}

    # Try to find rank 17
    if time.time() < deadline - 20:
        result17 = _try_structured_rank17(T, deadline - 20, n, m, p)
        if result17 is not None:
            U17, V17, W17 = result17
            if _check_decomposition(U17, V17, W17, T, tol=1e-8):
                return {"U": U17, "V": V17, "W": W17}

    # Try rank-revealing ALS for rank 17
    if time.time() < deadline - 15:
        result17 = _try_rank_revealing_als(T, 17, deadline - 15, n, m, p)
        if result17 is not None:
            U17, V17, W17 = result17
            if _check_decomposition(U17, V17, W17, T, tol=1e-8):
                return {"U": U17, "V": V17, "W": W17}

    # Try random ALS for rank 17
    if time.time() < deadline - 10:
        result17 = _try_random_als(T, 17, deadline - 10, n, m, p, n_seeds=8)
        if result17 is not None:
            U17, V17, W17 = result17
            if _check_decomposition(U17, V17, W17, T, tol=1e-8):
                return {"U": U17, "V": V17, "W": W17}

    # Try rank 16
    if time.time() < deadline - 8:
        result16 = _try_rank_revealing_als(T, 16, deadline - 8, n, m, p)
        if result16 is not None:
            U16, V16, W16 = result16
            if _check_decomposition(U16, V16, W16, T, tol=1e-8):
                return {"U": U16, "V": V16, "W": W16}

    if time.time() < deadline - 5:
        result16 = _try_random_als(T, 16, deadline - 5, n, m, p, n_seeds=5)
        if result16 is not None:
            U16, V16, W16 = result16
            if _check_decomposition(U16, V16, W16, T, tol=1e-8):
                return {"U": U16, "V": V16, "W": W16}

    # Return rank-18 solution
    return {"U": U18, "V": V18, "W": W18}
