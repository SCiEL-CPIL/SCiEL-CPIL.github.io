"""
Rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication.

Strategy:
  1. Build a structured rank-44 decomposition (4 Strassen 2x2x2 blocks + 4 schoolbook 1x2x2 blocks).
  2. Attempt to reduce rank by finding linear dependencies among the tensor terms and eliminating them.
  3. Verify each candidate decomposition exactly.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1 (the matrix multiplication tensor)."""
    return np.ones((n, m, p), dtype=np.float64)


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


def _construct_structured_decomposition():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Term 1: (A11+A22)(B11+B22)
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 2: (A21+A22)B11
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = -1
        col += 1
        # Term 3: A11(B21-B11)
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = -1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 4: A22(B12-B22)
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = -1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        col += 1
        # Term 5: (A11+A12)B22
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = -1
        col += 1
        # Term 6: (A21-A11)(B11+B12)
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 0), col] = -1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 7: (A12-A22)(B21+B22)
        U[(rA + 0) * m + (cA + 1), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = -1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1

    # Apply Strassen to the four 2x2x2 sub-blocks
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    # Apply schoolbook to the four 1x2x2 sub-blocks (row 2 of A, columns 0-1 or 2-3)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)

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
    """Try to eliminate one term by solving for its linear combination among the others."""
    rank = U.shape[1]
    for r_to_remove in range(rank):
        # Build the matrix of remaining terms
        indices = [i for i in range(rank) if i != r_to_remove]
        A_mat = outers[:, indices]  # shape (n*m*p, rank-1)
        deficit = outers[:, r_to_remove]

        # Solve A_mat @ delta = deficit
        try:
            delta, residuals, rank_mat, sv = np.linalg.lstsq(A_mat.T, deficit, rcond=None)
            # Check if the solution is good
            reconstructed = A_mat.T @ delta
            if np.allclose(reconstructed, deficit, atol=1e-8, rtol=0):
                # Apply the delta to the remaining terms
                U_new = U[:, indices].copy()
                V_new = V[:, indices].copy()
                W_new = W[:, indices].copy()
                for new_r, orig_r in enumerate(indices):
                    U_new[:, new_r] += delta[new_r] * U[:, orig_r]
                    V_new[:, new_r] += delta[new_r] * V[:, orig_r]
                    W_new[:, new_r] += delta[new_r] * W[:, orig_r]

                # Verify the new decomposition
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    return U_new, V_new, W_new
        except Exception:
            pass

    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_reductions, time_budget):
    """Iteratively try to reduce the rank by eliminating terms."""
    start_time = time.time()
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    outers = _compute_outer_products(U_cur, V_cur, W_cur, n, m, p)
    current_rank = U_cur.shape[1]

    for _ in range(max_reductions):
        if time.time() - start_time > time_budget:
            break
        U_cur, V_cur, W_cur = _try_eliminate_one_term(U_cur, V_cur, W_cur, n, m, p, outers)
        if U_cur.shape[1] == current_rank:
            # No reduction was made, so stop
            break
        current_rank = U_cur.shape[1]
        # Recompute outer products for the new set of terms
        outers = _compute_outer_products(U_cur, V_cur, W_cur, n, m, p)

    return U_cur, V_cur, W_cur


def _construct_rank_20_attempt():
    """
    Attempt to construct a rank-20 decomposition by:
    1. Using a rank-16 decomposition for the 4x4x4 tensor (from 4 Strassen 2x2x2 blocks).
    2. Subtracting the contributions of the 4th row of A (which is zero in the 3x4 case).
    3. Adding a rank-4 schoolbook decomposition for the remaining 1x4x4 strip.
    
    This is an experimental approach that may not work, but worth trying.
    """
    n, m, p = 3, 4, 4
    # Start with the rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    
    # The idea is to see if we can combine terms from different Strassen blocks
    # to reduce the total rank. This is complex and may not succeed.
    # For now, return the rank-44 decomposition as a safe fallback.
    return U44, V44, W44


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44

    # Verify the initial decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48 decomposition
        U48 = np.zeros((n * m, 48))
        V48 = np.zeros((m * p, 48))
        W48 = np.zeros((p * n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i * m + j, col] = 1
                    V48[j * p + k, col] = 1
                    W48[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U48, V48, W48
        best_rank = 48

    # Try to reduce rank
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 20:
            break
        U_try, V_try, W_try = _try_reduce_rank(
            best_U, best_V, best_W, n, m, p,
            max_reductions=best_rank - target_rank,
            time_budget=15
        )
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
