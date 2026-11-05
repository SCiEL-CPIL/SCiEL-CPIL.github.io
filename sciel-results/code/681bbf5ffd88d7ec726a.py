"""
Rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication.

Strategy:
  1. Partition the 3x4x4 tensor into:
     - Core: rows 0-1 of A, cols 0-1 of B, all 4 cols of C -> 2x2x4 (rank ~11 via 3x3x3 projection)
     - Actually: rows 0-2 of A, cols 0-1 of B, cols 0-3 of C -> 3x2x4
     - Strip1: rows 0-2 of A, cols 2-3 of B, cols 0-1 of C -> 3x2x2
     - Strip2: rows 0-2 of A, cols 2-3 of B, cols 2-3 of C -> 3x2x2
  2. Use rank-11 for 3x2x4 (from 3x3x3 rank-19 projection)
  3. Use rank-11 for each 3x2x2 (Strassen 7 for 2x2x2 + schoolbook 4 for 1x2x2)
  4. Total: 11 + 11 + 11 = 33... wait, that's too low. Let me recount.
  
  Actually: 3x2x4 core + two 3x2x2 strips.
  3x2x4: rank 11
  3x2x2: rank 11 each (7+4)
  Total: 11 + 11 + 11 = 33? No, that can't be right.
  
  Let me reconsider. The 3x4x4 tensor has dimensions (n=3, m=4, p=4).
  Partition by B's columns (m dimension):
    - B cols 0-1: A is 3x2, C is 4x3 -> 3x2x4 sub-tensor
    - B cols 2-3: A is 3x2, C is 4x3 -> 3x2x4 sub-tensor
  Wait, that gives two 3x2x4 sub-tensors, not one core + two strips.
  
  Let me try: partition A's rows:
    - A rows 0-1 (2x4): B is 4x4, C is 4x2 -> 2x4x4
    - A row 2 (1x4): B is 4x4, C is 4x1 -> 1x4x4
  
  2x4x4: partition B into 2x2 blocks: 2x2x4 + 2x2x4
  1x4x4: partition B into 2x2 blocks: 1x2x4 + 1x2x4
  
  This is getting complicated. Let me use the simpler approach from the current code
  (4 Strassen 2x2x2 + 4 schoolbook 1x2x2 = 28+16 = 44) and try to reduce.
  
  Actually, let me try the partition:
  - A rows 0-1, B cols 0-1, C cols 0-3: 2x2x4 (rank 7 via Strassen-like)
  - A rows 0-1, B cols 2-3, C cols 0-3: 2x2x4 (rank 7)
  - A row 2, B cols 0-1, C cols 0-3: 1x2x4 (rank 4)
  - A row 2, B cols 2-3, C cols 0-3: 1x2x4 (rank 4)
  Total: 7+7+4+4 = 22? That seems too low.
  
  Hmm, 2x2x4 matrix multiplication rank is 7 (Strassen for 2x2x2 gives 7, but 2x2x4 needs more).
  Actually 2x2x4: the rank is at most 8 (schoolbook) and at most 7+4=11 (Strassen 2x2x2 + schoolbook 2x2x2).
  The true rank of 2x2x4 is 7 (can be shown).
  
  Let me just use the current rank-44 decomposition and try harder to reduce it.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


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


def _construct_rank44():
    """Construct the rank-44 decomposition: 4 Strassen 2x2x2 + 4 schoolbook 1x2x2."""
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

    # Four 2x2x2 Strassen blocks
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    # Four 1x2x2 schoolbook blocks (row 2 of A)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)

    return U, V, W


def _construct_rank40():
    """
    Try to construct a rank-40 decomposition.
    
    Partition: 
    - A rows 0-1, B cols 0-3, C cols 0-3: 2x4x4
    - A row 2, B cols 0-3, C cols 0-3: 1x4x4
    
    2x4x4: partition B into two 2x2 blocks:
      - 2x2x4 (B cols 0-1, C cols 0-3): rank 7
      - 2x2x4 (B cols 2-3, C cols 0-3): rank 7
    1x4x4: partition B into two 2x2 blocks:
      - 1x2x4 (B cols 0-1, C cols 0-3): rank 4
      - 1x2x4 (B cols 2-3, C cols 0-3): rank 4
    
    Total: 7+7+4+4 = 22? That's way too low, so my rank estimates are wrong.
    
    Actually, 2x2x4 has rank at most 8 (schoolbook). The Strassen trick gives 7 for 2x2x2,
    but 2x2x4 needs more. Let me use schoolbook for 2x2x4 (rank 8) and 1x2x4 (rank 4).
    Total: 8+8+4+4 = 24. Still too low.
    
    I think the issue is that these sub-tensors share terms. Let me just use the rank-44
    and try to reduce via linear algebra.
    """
    return None


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
            # Use lstsq to find delta such that A_mat @ delta ≈ deficit
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


def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS

    # Start with rank-44 decomposition
    U44, V44, W44 = _construct_rank44()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44

    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48
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
    U_try, V_try, W_try = _try_reduce_rank(best_U, best_V, best_W, n, m, p, time_budget=120)
    if U_try.shape[1] < best_rank:
        best_U, best_V, best_W = U_try, V_try, W_try
        best_rank = U_try.shape[1]

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
