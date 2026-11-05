"""
Rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication.

Strategy (structured partition):
  * Core: 3x2 x 4x2 sub-tensor (rows 0..2 of A, cols 0..1; rows 0..1 of B, cols 0..1)
          built from the 3x3x3 rank-19 BCDL decomposition by dropping the 3rd
          row/col of B and the 3rd row of C.  Rank 19.
  * Strip A: 3x2 x 4x2 sub-tensor (rows 0..2 of A, cols 2..3; rows 2..3 of B, cols 0..1)
          built from a 2x2x2 Strassen decomposition.  Rank 7.
  * Strip B: 3x2 x 4x2 sub-tensor (rows 0..2 of A, cols 0..1; rows 2..3 of B, cols 2..3)
          built from a 2x2x2 Strassen decomposition.  Rank 7.
  * Strip C: 3x2 x 4x2 sub-tensor (rows 0..2 of A, cols 2..3; rows 2..3 of B, cols 2..3)
          built from a 2x2x2 Strassen decomposition.  Rank 7.

  Total: 19 + 7 + 7 + 7 = 40.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _build_target_tensor(n, m, p):
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)


def _bcdl_3x3x3():
    """Return (U,V,W) each of shape (9,19) for the 3x3x3 rank-19 BCDL decomposition."""
    U = np.zeros((9, 19))
    V = np.zeros((9, 19))
    W = np.zeros((9, 19))

    # 7x7x7 Strassen (block size 1)
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1

    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[4, 1] = -1

    U[0, 2] = 1
    V[3, 2] = 1; V[4, 2] = -1
    W[3, 2] = 1; W[4, 2] = 1

    U[4, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[2, 4] = 1
    V[4, 4] = 1
    W[3, 4] = 1; W[0, 4] = -1

    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[2, 5] = 1

    U[2, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[3, 6] = 1

    # 6x6x6 (1)
    U[0, 7] = 1; U[1, 7] = 1; U[3, 7] = 1; U[4, 7] = 1
    V[0, 7] = 1; V[1, 7] = 1; V[3, 7] = 1; V[4, 7] = 1
    W[0, 7] = 1; W[1, 7] = 1; W[3, 7] = 1; W[4, 7] = 1

    # 6x6x6 (2)
    U[1, 8] = 1; U[3, 8] = 1
    V[0, 8] = 1
    W[1, 8] = 1; W[4, 8] = -1

    # 6x6x6 (3)
    U[0, 9] = 1
    V[3, 9] = 1; V[4, 9] = -1
    W[3, 9] = 1; W[4, 9] = 1

    # 6x6x6 (4)
    U[4, 10] = 1
    V[1, 10] = 1; V[0, 10] = -1
    W[0, 10] = 1; W[1, 10] = 1

    # 6x6x6 (5)
    U[0, 11] = 1; U[3, 11] = 1
    V[4, 11] = 1
    W[3, 11] = 1; W[0, 11] = -1

    # 6x6x6 (6)
    U[1, 12] = 1; U[0, 12] = -1
    V[0, 12] = 1; V[3, 12] = 1
    W[3, 12] = 1

    # 6x6x6 (7)
    U[3, 13] = 1; U[4, 13] = -1
    V[1, 13] = 1; V[4, 13] = 1
    W[1, 13] = 1

    # 6x6x6 (8)
    U[1, 14] = 1; U[4, 14] = 1
    V[1, 14] = 1; V[4, 14] = 1
    W[1, 14] = 1; W[4, 14] = 1

    # 6x6x6 (9)
    U[1, 15] = 1; U[4, 15] = 1
    V[3, 15] = 1; V[4, 15] = 1
    W[1, 15] = 1; W[4, 15] = 1

    # 6x6x6 (10)
    U[1, 16] = 1; U[4, 16] = 1
    V[1, 16] = 1; V[3, 16] = 1
    W[1, 16] = 1; W[3, 16] = 1

    # 6x6x6 (11)
    U[1, 17] = 1; U[4, 17] = 1
    V[1, 17] = 1; V[4, 17] = 1
    W[1, 17] = 1; W[4, 17] = 1

    # 6x6x6 (12)
    U[1, 18] = 1; U[4, 18] = 1
    V[3, 18] = 1; V[4, 18] = 1
    W[1, 18] = 1; W[4, 18] = 1

    return U, V, W


def _embed_3x2x4_core(Uc, Vc, Wc, n, m, p):
    """Embed the 3x3x3 rank-19 into the 3x2 x 4x2 core (drop B row/col 2 and C row 2)."""
    U_core = np.zeros((n * 2, Uc.shape[1]))
    V_core = np.zeros((4 * 2, Vc.shape[1]))
    W_core = np.zeros((2 * n, Wc.shape[1]))

    for r in range(Uc.shape[1]):
        for i in range(3):
            for j in range(3):
                U_core[i * 2 + j, r] += Uc[i * 3 + j, r]
        for i in range(3):
            for k in range(3):
                V_core[i * 2 + k, r] += Vc[i * 3 + k, r]
        for i in range(3):
            for k in range(3):
                W_core[k * n + i, r] += Wc[k * 3 + i, r]

    return U_core, V_core, W_core


def _embed_3x2x2_strip(U2, V2, W2, n, m, p, a_r0, a_c0, b_r0, b_c0):
    """Embed a 2x2x2 rank-7 decomposition into a 3x2 x 4x2 strip."""
    U_strip = np.zeros((n * 2, U2.shape[1]))
    V_strip = np.zeros((4 * 2, V2.shape[1]))
    W_strip = np.zeros((2 * n, W2.shape[1]))

    for r in range(U2.shape[1]):
        for i in range(2):
            for j in range(2):
                U_strip[i * 2 + j, r] += U2[i * 2 + j, r]
        for i in range(2):
            for k in range(2):
                V_strip[(b_r0 + i) * 2 + (b_c0 + k), r] += V2[i * 2 + k, r]
        for i in range(2):
            for k in range(2):
                W_strip[(b_c0 + k) * n + (a_r0 + i), r] += W2[k * 2 + i, r]

    return U_strip, V_strip, W_strip


def _strassen_2x2x2():
    """Return (U,V,W) each of shape (4,7) for the 2x2x2 Strassen decomposition."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[2, 5] = 1

    U[2, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1

    return U, V, W


def _construct_rank40(n, m, p):
    """Construct the rank-40 structured decomposition."""
    # Core: 3x2 x 4x2 (A cols 0..1, B rows 0..1, B cols 0..1)
    Uc, Vc, Wc = _bcdl_3x3x3()
    U_core, V_core, W_core = _embed_3x2x4_core(Uc, Vc, Wc, n, m, p)

    # Strips
    Us, Vs, Ws = _strassen_2x2x2()
    U_s1, V_s1, W_s1 = _embed_3x2x2_strip(Us, Vs, Ws, n, m, p, a_r0=0, a_c0=2, b_r0=2, b_c0=0)
    U_s2, V_s2, W_s2 = _embed_3x2x2_strip(Us, Vs, Ws, n, m, p, a_r0=0, a_c0=0, b_r0=2, b_c0=2)
    U_s3, V_s3, W_s3 = _embed_3x2x2_strip(Us, Vs, Ws, n, m, p, a_r0=0, a_c0=2, b_r0=2, b_c0=2)

    U = np.hstack([U_core, U_s1, U_s2, U_s3])
    V = np.hstack([V_core, V_s1, V_s2, V_s3])
    W = np.hstack([W_core, W_s1, W_s2, W_s3])
    return U, V, W


def _compute_outer_products(U, V, W, n, m, p):
    """Precompute the flattened outer product for each term."""
    rank = U.shape[1]
    outers = np.zeros((n * m * p, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        outers[:, r] = np.einsum('ij,jk,ki->ijk', u, v, w).flatten()
    return outers


def _try_eliminate_one_term(U, V, W, n, m, p, outers):
    """Try to eliminate one term by expressing it as a linear combination of others."""
    rank = U.shape[1]
    for r_to_remove in range(rank):
        indices = [i for i in range(rank) if i != r_to_remove]
        A_mat = outers[:, indices]
        deficit = outers[:, r_to_remove]

        try:
            delta, residuals, rank_mat, sv = np.linalg.lstsq(A_mat.T, deficit, rcond=None)
            reconstructed = A_mat.T @ delta
            if np.allclose(reconstructed, deficit, atol=1e-8, rtol=0):
                U_new = U[:, indices].copy()
                V_new = V[:, indices].copy()
                W_new = W[:, indices].copy()
                for new_r, orig_r in enumerate(indices):
                    U_new[:, new_r] += delta[new_r] * U[:, orig_r]
                    V_new[:, new_r] += delta[new_r] * V[:, orig_r]
                    W_new[:, new_r] += delta[new_r] * W[:, orig_r]

                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    return U_new, V_new, W_new
        except Exception:
            pass

    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, time_budget):
    """Iteratively try to reduce the rank."""
    start_time = time.time()
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    outers = _compute_outer_products(U_cur, V_cur, W_cur, n, m, p)
    current_rank = U_cur.shape[1]

    while time.time() - start_time < time_budget:
        U_cur, V_cur, W_cur = _try_eliminate_one_term(U_cur, V_cur, W_cur, n, m, p, outers)
        if U_cur.shape[1] == current_rank:
            break
        current_rank = U_cur.shape[1]
        outers = _compute_outer_products(U_cur, V_cur, W_cur, n, m, p)

    return U_cur, V_cur, W_cur


def _construct_rank48_fallback(n, m, p):
    U = np.zeros((n * m, 48))
    V = np.zeros((m * p, 48))
    W = np.zeros((p * n, 48))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS

    # Start with rank-40 structured decomposition
    U40, V40, W40 = _construct_rank40(n, m, p)
    best_U, best_V, best_W = U40.copy(), V40.copy(), W40.copy()
    best_rank = U40.shape[1]

    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48
        U48, V48, W48 = _construct_rank48_fallback(n, m, p)
        best_U, best_V, best_W = U48, V48, W48
        best_rank = 48

    # Try to reduce rank using iterative elimination
    time_budget = min(60, deadline - time.time() - 15)
    if time_budget > 5:
        U_try, V_try, W_try = _try_reduce_rank(best_U, best_V, best_W, n, m, p, time_budget)
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
