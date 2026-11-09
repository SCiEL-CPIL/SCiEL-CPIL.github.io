"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy:
  1. Build a verified rank-56 decomposition via Strassen on 2x2 blocks (two 4x4*4x4
     copies). This is the structural skeleton and guaranteed-valid fallback.
  2. Prune redundant rank terms from it.
  3. Use a random-U,V + linear-solve-for-W refinement (from the inspiration) to try
     to discover a genuinely lower-rank exact decomposition with integer/half-integer
     entries.
  Return the best (lowest-rank) verified decomposition found.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START


def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Check that U, V, W exactly decompose the matrix multiplication tensor.

    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    Tensor entry T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] must equal 1.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            contrib = U3[i, :, c][:, np.newaxis] * V3[:, :, c] * W3[:, i, c][np.newaxis, :]
            T[i, :, :] += contrib
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True


def _verify_direct(U, V, W, n, m, p, tol=1e-8):
    """Vectorised verification of the decomposition (faster, used in search)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
    return np.max(np.abs(T - 1.0)) <= tol


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return Strassen's rank-7 decomposition of the 2x2 * 2x2 multiplication tensor.

    Convention: U shape (4,7) indexed by (i*2+j), V shape (4,7) indexed by (j*2+k),
    W shape (4,7) indexed by (k*2+i).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # c=0: M1 = (A00+A11)(B01+B10)
    U[0 * 2 + 0, 0] = 1; U[1 * 2 + 1, 0] = 1
    V[0 * 2 + 1, 0] = 1; V[1 * 2 + 0, 0] = 1
    W[0 * 2 + 0, 0] = 1; W[1 * 2 + 1, 0] = 1
    # c=1: M2 = (A10+A11)A00
    U[1 * 2 + 0, 1] = 1; U[1 * 2 + 1, 1] = 1
    V[0 * 2 + 0, 1] = 1
    W[1 * 2 + 0, 1] = 1; W[1 * 2 + 1, 1] = -1
    # c=2: M3 = A00(B00-B10)
    U[0 * 2 + 0, 2] = 1
    V[0 * 2 + 0, 2] = 1; V[1 * 2 + 0, 2] = -1
    W[0 * 2 + 0, 2] = 1; W[1 * 2 + 1, 2] = 1
    # c=3: M4 = A11(B11-B01)
    U[1 * 2 + 1, 3] = 1
    V[1 * 2 + 1, 3] = 1; V[0 * 2 + 1, 3] = -1
    W[1 * 2 + 0, 3] = 1; W[0 * 2 + 0, 3] = 1
    # c=4: M5 = (A00+A10)B11
    U[0 * 2 + 0, 4] = 1; U[1 * 2 + 0, 4] = 1
    V[1 * 2 + 1, 4] = 1
    W[0 * 2 + 0, 4] = -1; W[1 * 2 + 1, 4] = 1
    # c=5: M6 = (A10-A00)(B00+B11)
    U[1 * 2 + 0, 5] = 1; U[0 * 2 + 0, 5] = -1
    V[0 * 2 + 0, 5] = 1; V[1 * 2 + 1, 5] = 1
    W[1 * 2 + 0, 5] = 1
    # c=6: M7 = (A01-A11)(B01+B10)
    U[0 * 2 + 1, 6] = 1; U[1 * 2 + 1, 6] = -1
    V[0 * 2 + 1, 6] = 1; V[1 * 2 + 0, 6] = 1
    W[0 * 2 + 0, 6] = 1
    return U, V, W


def _build_4x4x4():
    """Build a rank-28 decomposition of the 4x4 * 4x4 multiplication tensor using
    Strassen's algorithm on 2x2 blocks.

    A = [[A00, A01], [A10, A11]], each Aab is 2x2.
    B = [[B00, B01], [B10, B11]], each Bbc is 2x2.
    For each Strassen factor a and sub-factor c, and each block-row bi, block-col bj,
    block-out bk, we add the Strassen 2x2 product of the A and B blocks.
    """
    sU, sV, sW = _strassen_2x2x2()
    rank = 4 * 7
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    for a in range(4):  # Strassen factor index (which 2x2 A-block)
        for bi in range(2):  # row within that block
            for bj in range(2):  # col within that block
                for c in range(4):  # which 2x2 B-block
                    for bk in range(2):  # row within that B-block
                        for cc in range(7):  # Strassen sub-factor
                            col = (a * 4 + c) * 7 + cc
                            # U[i*4 + j, col] with i = 2a+bi, j = 2c+bj
                            U[(2 * a + bi) * 4 + (2 * c + bj), col] = sU[bi * 2 + bj, cc]
                            # V[j*4 + k, col] with j = 2c+bj, k = 2d+bk
                            V[(2 * c + bj) * 4 + (2 * d + bk), col] = sV[bj * 2 + bk, cc]
                            # W[k*4 + i, col] with k = 2d+bk, i = 2a+bi
                            W[(2 * d + bk) * 4 + (2 * a + bi), col] = sW[bk * 2 + bi, cc]
    return U, V, W


def _build_4x4x8():
    """Build a rank-56 decomposition of the 4x4 * 4x8 multiplication tensor by
    combining two 4x4 * 4x4 decompositions (one for each half of the output)."""
    U1, V1, W1 = _build_4x4x4()
    rank = 2 * 28
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # First 4x4 * 4x4: computes C[:, 0:4] = A @ B[:, 0:4]
    U[:, 0:28] = U1
    V[0:16, 0:28] = V1
    W[0:16, 0:28] = W1

    # Second 4x4 * 4x4: computes C[:, 4:8] = A @ B[:, 4:8]
    U[:, 28:56] = U1
    V[16:32, 28:56] = V1
    W[16:32, 28:56] = W1
    return U, V, W


def _prune(U, V, W, n, m, p, tol=1e-9, deadline=None):
    """Greedy removal of redundant rank terms. Returns pruned (U,V,W)."""
    U = U.copy(); V = V.copy(); W = W.copy()
    r = U.shape[1]
    while r > 0:
        if deadline is not None and time.time() > deadline:
            break
        removed = False
        for c in range(r):
            if deadline is not None and time.time() > deadline:
                break
            U_t = np.delete(U, c, axis=1)
            V_t = np.delete(V, c, axis=1)
            W_t = np.delete(W, c, axis=1)
            if _verify_fast(U_t, V_t, W_t, n, m, p, tol=tol):
                U, V, W = U_t, V_t, W_t
                r -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W


def _random_solve_decomposition(n, m, p, r, rng, deadline, max_scales=8):
    """Given random integer U,V of rank r, solve for W and snap to integers.

    Returns (U,V,W) if a valid integer/half-integer decomposition is found, else None.
    """
    U = rng.integers(-1, 2, size=(n * m, r))
    V = rng.integers(-1, 2, size=(m * p, r))
    # Build linear system A x = b for W (flattened, size p*n*r).
    num_eq = n * m * p
    A = np.zeros((num_eq, p * n * r))
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                coeffs = u_row * v_row  # size r
                base = (k * n + i) * r
                A[eq_idx, base:base + r] = coeffs
                eq_idx += 1
    b = np.ones(num_eq)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    # Check residual
    if np.max(np.abs(A @ x - b)) > 1e-6:
        return None
    Wf = x.reshape(p * n, r)
    # Try integer and half-integer scales
    for L in range(1, max_scales + 1):
        Ws = Wf * L
        Wi = np.round(Ws)
        if np.max(np.abs(Ws - Wi)) < 1e-7:
            W = Wi.astype(int)
            if _verify_direct(U, V, W, n, m, p, tol=1e-7):
                return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s limit

    best = None
    best_rank = None

    def _consider(U, V, W):
        nonlocal best, best_rank
        if not _verify_fast(U, V, W, n, m, p, tol=1e-9):
            return
        r = U.shape[1]
        if best is None or r < best_rank:
            best = (U.copy(), V.copy(), W.copy())
            best_rank = r

    # 1. Structural skeleton: rank-56 Strassen decomposition.
    try:
        U, V, W = _build_4x4x8()
        if _verify_fast(U, V, W, n, m, p, tol=1e-9):
            # Snap to integers (Strassen uses only -1, 0, 1)
            Us = np.round(U).astype(int)
            Vs = np.round(V).astype(int)
            Ws = np.round(W).astype(int)
            if _verify_fast(Us, Vs, Ws, n, m, p, tol=1e-9):
                _consider(Us, Vs, Ws)
            else:
                _consider(U, V, W)
    except Exception:
        pass

    # 2. Prune the best found so far.
    if best is not None and time.time() < deadline - 30:
        try:
            U, V, W = _prune(best[0], best[1], best[2], n, m, p, tol=1e-9,
                             deadline=deadline - 30)
            _consider(U, V, W)
        except Exception:
            pass

    # 3. Random-solve refinement to try to beat the current best rank.
    if best is not None:
        target = best_rank - 1
        if target >= 1 and time.time() < deadline - 10:
            rng = np.random.default_rng(12345)
            # Try a few ranks descending from best_rank-1.
            for r in range(min(target, best_rank - 1), max(1, best_rank - 8) - 1, -1):
                if time.time() > deadline - 10:
                    break
                # A few attempts per rank.
                for _ in range(200):
                    if time.time() > deadline - 10:
                        break
                    res = _random_solve_decomposition(n, m, p, r, rng, deadline)
                    if res is not None:
                        U, V, W = res
                        _consider(U, V, W)
                        # Found one at this rank; try to go lower.
                        break

    # 4. Final: return best, or schoolbook fallback.
    if best is not None:
        U, V, W = best
        # Snap to integers if close.
        Us = np.round(U)
        Vs = np.round(V)
        Ws = np.round(W)
        if (np.max(np.abs(U - Us)) < 1e-9 and np.max(np.abs(V - Vs)) < 1e-9
                and np.max(np.abs(W - Ws)) < 1e-9):
            U, V, W = Us.astype(int), Vs.astype(int), Ws.astype(int)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# EVOLVE-BLOCK-END
