"""Construct a rank-minimal tensor decomposition for 3x3 * 3x4 matrix multiplication.

Starts from a rank-34 Strassen-based construction, then searches for a lower-rank
exact decomposition by applying random invertible basis transformations to the
three factor matrices. Under the transformation
    U' = U @ A,  V' = V @ B,  W' = W @ (A ⊗ B)^{-1}
the tensor is preserved (rank invariant), but a different basis may expose
linearly dependent / zero columns that can be dropped, revealing a lower-rank
exact decomposition.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def _verify(n, m, p, U, V, W):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    r = len(U[0])
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(Ua[i*m+j, rr] * Va[j*p+k, rr] * Wa[k*n+i, rr] for rr in range(r))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _build_rank34_strassen():
    """Rank-34 decomposition: 2 Strassen blocks (2x2x2) + schoolbook for remaining parts.

    Split the 3x3x4 multiplication into:
      * Top-left 2x2x2 block (i=0,1; j=0,1; k=0,1): Strassen rank 7
      * Top-right 2x2x2 block (i=0,1; j=0,1; k=2,3): Strassen rank 7
      * Bottom row (i=2; j=0,1,2; k=0,1,2,3): schoolbook rank 12
      * Right column (i=0,1; j=2; k=0,1,2,3): schoolbook rank 8
    Total: 7 + 7 + 12 + 8 = 34
    """
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Term 1: M1 = (A11+A22) * (B11+B22) -> C11 += M1, C22 += M1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1

        # Term 2: M2 = (A21+A22) * B11 -> C21 += M2, C22 -= M2
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1

        # Term 3: M3 = A11 * (B12-B22) -> C12 += M3, C22 += M3
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1

        # Term 4: M4 = A22 * (B21-B11) -> C11 += M4, C21 += M4
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1

        # Term 5: M5 = (A11+A12) * B22 -> C11 -= M5, C12 += M5
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1

        # Term 6: M6 = (A21-A11) * (B11+B12) -> C22 += M6
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1

        # Term 7: M7 = (A12-A22) * (B21+B22) -> C11 += M7
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1

    # 1. Top-left 2x2x2 block (i=0,1; j=0,1; k=0,1)
    add_strassen(0, 0, 0)
    # 2. Top-right 2x2x2 block (i=0,1; j=0,1; k=2,3)
    add_strassen(0, 0, 2)
    # 3. Bottom row (i=2; j=0,1,2; k=0,1,2,3)
    for j in range(3):
        for k in range(4):
            add_term(2 * m + j, j * p + k, k * n + 2)
    # 4. Right column (i=0,1; j=2; k=0,1,2,3)
    for i in range(2):
        for k in range(4):
            add_term(i * m + 2, 2 * p + k, k * n + i)

    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Schoolbook rank-36 decomposition (fallback)."""
    n, m, p = 3, 3, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_transform(U, V, W, A, B, tol=1e-10):
    """Apply basis transform U'=U@A, V'=V@B, W'=W@(A⊗B)^{-1}; drop zero columns.

    Returns (U', V', W') as float arrays with zero columns removed, or None if
    the transform is numerically singular or fails to preserve the tensor.
    """
    r = U.shape[1]
    try:
        # A ⊗ B  (Kronecker), shape (r*r, r*r)
        Kr = np.kron(A, B)
        # W' = W @ Kr^{-1}
        Wt = W @ np.linalg.inv(Kr)
    except np.linalg.LinAlgError:
        return None

    Ut = U @ A
    Vt = V @ B

    # Detect zero columns across all three factors simultaneously
    keep = []
    for c in range(r):
        if (np.max(np.abs(Ut[:, c])) > tol and
                np.max(np.abs(Vt[:, c])) > tol and
                np.max(np.abs(Wt[:, c])) > tol):
            keep.append(c)
    keep = np.array(keep, dtype=int)

    return Ut[:, keep], Vt[:, keep], Wt[:, keep]


def _search_lower_rank(base, n, m, p, deadline, seed=12345):
    """Search for a lower-rank exact decomposition via random basis transforms."""
    rng = np.random.default_rng(seed)
    U0 = np.asarray(base["U"], dtype=float)
    V0 = np.asarray(base["V"], dtype=float)
    W0 = np.asarray(base["W"], dtype=float)
    r0 = U0.shape[1]

    best = (r0, base)  # (rank, result_dict)

    # Try several random invertible transforms of increasing "aggressiveness".
    # A = I + alpha * G where G is random; for small alpha near-identity (refinement),
    # for larger alpha more exploratory.
    scales = [0.1, 0.3, 1.0, 3.0, 10.0]
    for s in scales:
        if time.time() > deadline - 5:
            break
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            G1 = rng.standard_normal((r0, r0))
            G2 = rng.standard_normal((r0, r0))
            A = np.eye(r0) + s * G1
            B = np.eye(r0) + s * G2
            res = _try_transform(U0, V0, W0, A, B)
            if res is None:
                continue
            Ut, Vt, Wt = res
            r_new = Ut.shape[1]
            if r_new >= best[0]:
                continue
            # Verify exactness of the reduced decomposition
            cand = {"U": Ut.tolist(), "V": Vt.tolist(), "W": Wt.tolist()}
            if _verify(n, m, p, Ut.tolist(), Vt.tolist(), Wt.tolist()):
                best = (r_new, cand)
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before the 150s kill

    n, m, p = 3, 3, 4

    # Start from the rank-34 Strassen-based decomposition.
    base = _build_rank34_strassen()
    if not _verify(n, m, p, base["U"], base["V"], base["W"]):
        # Fall back to schoolbook if the construction is broken.
        base = _build_schoolbook()
        return base

    # Search for a lower-rank exact decomposition via random basis transforms.
    try:
        best_rank, best = _search_lower_rank(base, n, m, p, deadline)
        return best
    except Exception:
        return base
# EVOLVE-BLOCK-END
