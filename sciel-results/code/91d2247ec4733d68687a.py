"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_strassen_schoolbook(n, m, p):
    """Build the known rank-50 decomposition: 6 Strassen (2x2x2) + 2 schoolbook (2x2x1)."""
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # Strassen 7 multiplications for 2x2x2
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        # M5 = (a-b)(h+g)
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Apply Strassen to 2x2x2 blocks: (0,0,0), (0,1,0), (0,2,0)
    # Then (0,0,1), (0,1,1), (0,2,1)
    # Then (0,0,2), (0,1,2), (0,2,2)
    # But we only have m=4, p=7. So j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    # Strassen blocks cover (j_start, k_start) to (j_start+1, k_start+1)
    # j_start in {0, 2}, k_start in {0, 2, 4} gives 6 Strassen blocks
    # Remaining: j_start=0, k_start=6 and j_start=2, k_start=6 are 2x2x1 blocks

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    return U, V, W, col

def _verify(U, V, W, n, m, p):
    """Verify the decomposition reproduces the all-ones tensor using matrix products."""
    r = U.shape[1]
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    T_check = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
    expected = np.ones((n, m, p), dtype=np.int64)
    return np.array_equal(T_check, expected)

def _check_pairwise_cancellation(U, V, W, n, m, p):
    """
    Check if any pair of columns can be removed by finding a pair r1, r2 such that
    the sum of their rank-1 tensors is zero. This requires U[:,r1] ⊗ V[:,r1] ⊗ W[:,r1] = -U[:,r2] ⊗ V[:,r2] ⊗ W[:,r2].
    For integer entries, this means the supports must be identical and the products must be exact negatives.
    """
    r = U.shape[1]
    for r1 in range(r):
        for r2 in range(r1 + 1, r):
            # Check if the tensor contributions are exact negatives
            # T_r1[i,j,k] = U[i,j,r1] * V[j,k,r1] * W[k,i,r1]
            # T_r2[i,j,k] = U[i,j,r2] * V[j,k,r2] * W[k,i,r2]
            # We need T_r1 + T_r2 = 0
            # This is expensive to check fully, but for small sizes we can check a few key entries
            # Actually, for exact cancellation, the non-zero patterns must match
            u1 = U[:, r1]
            u2 = U[:, r2]
            # Quick check: if the non-zero positions differ, skip
            nz1 = np.nonzero(u1)[0]
            nz2 = np.nonzero(u2)[0]
            if len(nz1) != len(nz2):
                continue
            # For the actual tensor product, we need to check all (i,j,k)
            # But this is expensive. Instead, we can just try removing the pair and verifying
            # But that's what the main search does. So this function is just a placeholder for a fast check.
            # A faster check: if U[:,r1] = -c * U[:,r2] for some c, and similarly for V and W
            # But the scaling must be consistent.
            pass
    return []

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    n, m, p = DIMENSIONS

    # Build the known rank-50 decomposition
    U, V, W, col = _build_strassen_schoolbook(n, m, p)
    rank = col

    # Verify the rank-50 decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook (should not happen)
        return _schoolbook(n, m, p)

    # Remove zero columns if any
    zero_mask = (np.all(U == 0, axis=0) | np.all(V == 0, axis=0) | np.all(W == 0, axis=0))
    if zero_mask.any():
        keep = ~zero_mask
        U, V, W = U[:, keep], V[:, keep], W[:, keep]
        rank = U.shape[1]

    # Try to remove single columns (fast check)
    if time.time() < deadline - 30:
        U_work, V_work, W_work = U.copy(), V.copy(), W.copy()
        rank_work = U_work.shape[1]
        improved = True
        while improved and time.time() < deadline - 25:
            improved = False
            for r in range(rank_work):
                mask = np.ones(rank_work, dtype=bool)
                mask[r] = False
                U_new = U_work[:, mask]
                V_new = V_work[:, mask]
                W_new = W_work[:, mask]
                if _verify(U_new, V_new, W_new, n, m, p):
                    U_work, V_work, W_work = U_new, V_new, W_new
                    rank_work = U_work.shape[1]
                    improved = True
                    U, V, W = U_work, V_work, W_work
                    rank = rank_work
                    break

    # Try to remove pairs of columns
    if time.time() < deadline - 20:
        U_try, V_try, W_try = U.copy(), V.copy(), W.copy()
        rank_try = U_try.shape[1]
        for r1 in range(rank_try):
            for r2 in range(r1 + 1, rank_try):
                if time.time() > deadline - 15:
                    break
                mask = np.ones(rank_try, dtype=bool)
                mask[r1] = False
                mask[r2] = False
                U_new = U_try[:, mask]
                V_new = V_try[:, mask]
                W_new = W_try[:, mask]
                if _verify(U_new, V_new, W_new, n, m, p):
                    U, V, W = U_new, V_new, W_new
                    rank = U_new.shape[1]
                    U_try, V_try, W_try = U_new, V_new, W_new
                    rank_try = U_new.shape[1]
                    break
            if time.time() > deadline - 15:
                break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
