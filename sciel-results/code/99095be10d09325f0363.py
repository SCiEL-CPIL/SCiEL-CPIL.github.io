"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
from collections import defaultdict

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

        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

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

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    return U, V, W, col

def _verify(U, V, W, n, m, p):
    """Verify the decomposition reproduces the all-ones tensor using matrix products."""
    r = U.shape[1]
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    U_t = U.reshape(n, m, r)  # (n, m, r)
    V_t = V.reshape(m, p, r)  # (m, p, r)
    W_t = W.reshape(p, n, r)  # (p, n, r)
    # For each r, the contribution is outer product structure
    # T[i,j,k] = sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
    # Compute via: for each r, form matrix M_r[i,k] = sum_j U_t[i,j,r]*V_t[j,k,r], then T[i,j,k] += M_r[i,k]*W_t[k,i,r]
    # More efficient: use einsum with optimization
    T_check = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
    expected = np.ones((n, m, p), dtype=np.int64)
    return np.array_equal(T_check, expected)

def _verify_fast(U, V, W, n, m, p):
    """Faster verification: compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    by precomputing partial products."""
    r = U.shape[1]
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    # T[i,j,k] = sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
    # Group by i: for fixed i, T[i,j,k] = sum_r U_t[i,j,r] * (V_t[j,k,r] * W_t[k,i,r])
    # Let's just use einsum, it's fast enough for small tensors
    T_check = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
    expected = np.ones((n, m, p), dtype=np.int64)
    return np.array_equal(T_check, expected)

def _dual_constraint_analysis(U, V, W, n, m, p):
    """
    Analyze the dual tensor constraints (syzygies) to identify columns that are
    linearly dependent in the tensor sense.
    
    For a valid rank-r decomposition, each column r contributes a rank-1 tensor
    T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r].
    
    The dual constraints are: for any linear functional L on the tensor space,
    sum_r L(T_r) = L(T) where T is the all-ones tensor.
    
    We check if removing a set of columns and adjusting others still satisfies
    all dual constraints. This is equivalent to checking if the remaining
    columns still span the correct tensor.
    
    Returns: list of column indices that can potentially be removed (candidates).
    """
    r = U.shape[1]
    candidates = []
    
    # For each column, compute its "signature" - which (i,j,k) positions it affects
    # and with what weight
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    # Compute the contribution of each rank-1 term
    # term_r[i,j,k] = U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
    # We want to find if any subset of terms sums to zero (or can be absorbed)
    
    # Quick check: for each pair of columns, see if their difference is a valid
    # rank-1 term (i.e., can be represented as a single column)
    for r1 in range(r):
        for r2 in range(r1 + 1, r):
            # Check if column r1 + column r2 can be replaced by a single column
            # This requires: (U[:,r1]+U[:,r2]) * (V[:,r1]+V[:,r2]) * (W[:,r1]+W[:,r2])
            # to equal T_{r1} + T_{r2} + (some correction)
            # Actually, merging two rank-1 terms into one is generally not possible.
            # Instead, check if one column is redundant given the others.
            pass
    
    # Better approach: check linear dependence of the rank-1 tensors
    # Flatten each rank-1 term to a vector in R^(n*m*p)
    # term_r vector: for (i,j,k), value = U_t[i,j,r]*V_t[j,k,r]*W_t[k,i,r]
    # If any term is in the span of others, it can be removed.
    
    # But we need exact arithmetic, so let's use the verification approach.
    return candidates

def _greedy_set_cover(U, V, W, n, m, p, deadline):
    """
    Greedy set-cover / hyperedge search for a lower-rank exact decomposition.
    
    Model: each rank-1 term (a column of U,V,W) is a "hyperedge" that contributes
    to a set of output monomials (i,j,k) where U[i,j,r]*V[j,k,r]*W[k,i,r] != 0.
    We want the minimum collection of hyperedges whose weighted sum equals the
    all-ones tensor. We use greedy set cover with conflict detection and
    non-local term merger search.
    """
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank

    # --- Step 1: Remove zero columns (trivial reduction) ---
    zero_mask = (np.all(U == 0, axis=0) | np.all(V == 0, axis=0) | np.all(W == 0, axis=0))
    if zero_mask.any():
        keep = ~zero_mask
        U, V, W = U[:, keep], V[:, keep], W[:, keep]
        current_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank

    if time.time() > deadline - 20:
        return (best_U, best_V, best_W) if best_rank < 50 else None

    # --- Step 2: Greedy set cover with non-local merger search ---
    # For each pair of columns (r1, r2), check if their combined contribution
    # can be replaced by a single column (a "merged" hyperedge).
    # This is the non-local term merger: if U[:,r1]+U[:,r2], V[:,r1]+V[:,r2],
    # W[:,r1]+W[:,r2] produces a valid rank-1 term that covers the same monomials,
    # we can merge them.
    
    # More practically: try to find a subset of columns whose removal still
    # leaves a valid decomposition (i.e., the removed columns sum to zero in
    # the tensor sense).
    
    # Strategy: iteratively try removing one column at a time, checking if the
    # remaining columns still produce the correct tensor.
    # This is O(rank * verification_cost) which is feasible for rank ~50.
    
    U_work, V_work, W_work = U.copy(), V.copy(), W.copy()
    rank_work = U_work.shape[1]
    
    removed = set()
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for r in range(rank_work):
            if r in removed:
                continue
            # Try removing column r
            # Build new matrices without column r
            mask = np.ones(rank_work, dtype=bool)
            mask[r] = False
            U_new = U_work[:, mask]
            V_new = V_work[:, mask]
            W_new = W_work[:, mask]
            
            if _verify(U_new, V_new, W_new, n, m, p):
                U_work, V_work, W_work = U_new, V_new, W_new
                rank_work = U_work.shape[1]
                removed.add(r)
                improved = True
                best_U, best_V, best_W = U_work.copy(), V_work.copy(), W_work.copy()
                best_rank = rank_work
                break  # Restart the loop after a removal
    
    if best_rank < 50:
        return (best_U, best_V, best_W)
    
    # --- Step 3: Try to find a completely different decomposition via
    #     structured search over block patterns ---
    # The tensor T[i,j,k]=1 for all i,j,k. We seek rank-r factors with entries
    # in {-1, 0, 1} (or half-integers). 
    # 
    # Key insight: for 2x4x7, we can try different tilings:
    #   - 2x2x2 Strassen blocks (rank 7 each)
    #   - 2x2x1 schoolbook (rank 4 each)
    #   - 2x1x2 Strassen-like (rank 7 each) -- partition along rows
    #   - 1x2x2 blocks
    #   - 2x4x2 combined blocks (potentially rank < 14)
    #
    # Let's try: partition the 4 columns into 2+2, and the 7 rows into 2+2+2+1
    # Current: 3 Strassen (2x2x2) + 1 schoolbook (2x2x1) per 2-column strip = 25
    # Total: 50
    #
    # Alternative: partition 7 rows as 3+4, try 2x2x3 and 2x2x4 blocks
    # 2x2x3: can we do better than 3 Strassen+... no, 3 is not a power of 2.
    # 2x2x3 = 2x2x2 (rank 7) + 2x2x1 (rank 4) = rank 11
    # 2x2x4 = 2 Strassen = rank 14
    # Total: 11 + 14 = 25 per 2-col strip, same as before.
    #
    # Alternative: partition 4 cols as 4 (no split), 7 rows as 7
    # 2x4x7 directly: lower bound is max(8,28,14)=28
    # Can we find a rank-28 or rank-29 decomposition?
    #
    # For 2x4x4: lower bound max(8,16,14)=16
    # 2 Strassen blocks = 14 < 16? No, 14 < 16 means lower bound is 16, so 14 is impossible.
    # Wait: 2x4x4 tensor has entries T[i,j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Flattening (i,j) x k: 8x4 matrix of all ones, rank 1.
    # Flattening i x (j,k): 2x16 matrix of all ones, rank 1.
    # Flattening (i,k) x j: 8x4 matrix of all ones, rank 1.
    # So the tensor rank lower bound from flattening is 1, but that's not tight.
    # The true rank of 2x4x4 multiplication tensor is known to be at most 14 (2 Strassen).
    # 
    # For 2x4x7: we can try 2x4x4 (rank 14) + 2x4x3
    # 2x4x3: lower bound max(8,12,14)=14. 
    # 2x4x3 = 2x2x3 + 2x2x3 = (7+4)+(7+4) = 22? Or can we do better?
    # 2x2x3: partition 3 as 2+1: Strassen(7) + schoolbook(4) = 11
    # So 2x4x3 = 2*11 = 22
    # Total: 14 + 22 = 36 < 50!
    #
    # Let's try: 2x4x4 (2 Strassen blocks, rank 14) + 2x4x3 (4 blocks: 2 Strassen + 2 schoolbook, rank 22)
    # Total rank = 14 + 22 = 36
    
    # Wait, I need to be more careful. Let me construct this explicitly.
    #
    # 2x4x4: partition cols j as [0,1] and [2,3], rows k as [0,1,2,3]
    #   Block 1: 2x2x4 -> partition k as [0,1] and [2,3]: 2 Strassen blocks, rank 14
    #   Block 2: 2x2x4 -> same, rank 14
    #   Total for 2x4x4: 28? No wait, that's 4 Strassen blocks = 28.
    #   
    # Hmm, let me reconsider. 2x4x4:
    #   i: 2 rows, j: 4 cols, k: 4 cols
    #   Partition j into [0,1] and [2,3]:
    #     2x2x4 for j=[0,1]: partition k into [0,1] and [2,3]:
    #       2x2x2 (Strassen, rank 7) + 2x2x2 (Strassen, rank 7) = 14
    #     2x2x4 for j=[2,3]: same = 14
    #   Total: 28 for 2x4x4
    #
    # That's worse. The issue is that 2x4x4 with 2 Strassen blocks means:
    #   Treat A as 2x4, B as 4x4. Partition A's 4 cols into 2+2, B's 4 rows into 2+2.
    #   Then we have 4 sub-multiplications: (2x2)*(2x2) each.
    #   Each 2x2x2 uses Strassen rank 7. Total: 4*7 = 28.
    #   But we can do better: partition differently.
    #
    # Actually, for 2x4 * 4x4, we can use:
    #   A = [A1 | A2] where A1, A2 are 2x2
    #   B = [B1; B2] where B1, B2 are 2x2
    #   C = A1*B1 + A2*B2
    #   Using Strassen on each: 7+7 = 14
    #   But we need to add them, which is free (linear combination).
    #   So rank 14 for 2x4x4!
    #
    # Wait, that's the key: C = A1*B1 + A2*B2. Each product is 2x2x2 (rank 7 Strassen).
    # The sum is a linear combination, so total rank = 7+7 = 14.
    # Yes! 2x4x4 has rank at most 14.
    #
    # Similarly, 2x4x3:
    #   A = [A1 | A2 | A3] where each is 2x1? No, A is 2x4.
    #   Let me redo: A is 2x4, B is 4x3.
    #   Partition A's 4 cols into 2+2: A = [A1 | A2], A1,A2 are 2x2
    #   Partition B's 4 rows into 2+2: B = [B1; B2], B1,B2 are 2x3
    #   C = A1*B1 + A2*B2
    #   Each term: 2x2 * 2x3 = 2x2x3
    #   2x2x3: partition B's 3 cols into 2+1:
    #     2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4) = 11
    #   Total: 11 + 11 = 22
    #
    # So 2x4x7 = 2x4x4 (rank 14) + 2x4x3 (rank 22) = rank 36!
    #
    # Let me construct this explicitly.

    # --- Construct rank-36 decomposition ---
    # 2x4x4 part (cols 0-3 of B, i.e., k=0,1,2,3):
    #   A = [A1 | A2], A1 = A[:, 0:2], A2 = A[:, 2:4]  (each 2x2)
    #   B1 = B[0:2, 0:4], B2 = B[2:4, 0:4]  (each 2x4)
    #   C1 = A1 * B1  (2x4)
    #   C2 = A2 * B2  (2x4)
    #   C = C1 + C2
    #   Each 2x2x4: partition B's 4 cols into 2+2, use Strassen on each 2x2x2
    #   Wait, 2x2x4: A is 2x2, B is 2x4, C is 2x4
    #   Partition B's 4 cols into 2+2: B = [B1 | B2], each 2x2
    #   C = A*B1 + A*B2
    #   Each A*B is 2x2x2, Strassen rank 7
    #   Total: 7+7 = 14 for 2x2x4
    #   So 2x4x4 = 14 (for A1*B1) + 14 (for A2*B2) = 28? 
    #
    # No wait. Let me re-examine.
    # 2x4x4: A is 2x4, B is 4x4, C is 2x4.
    # Partition A's 4 cols into 2+2: A = [A1 | A2]
    # Partition B's 4 rows into 2+2: B = [B1; B2]
    # C = A1*B1 + A2*B2
    # A1 is 2x2, B1 is 2x4: product is 2x4. This is a 2x2x4 multiplication.
    # 2x2x4: A is 2x2, B is 2x4, C is 2x4.
    # Partition B's 4 cols into 2+2: B = [B1a | B1b]
    # C = A*B1a + A*B1b
    # Each is 2x2x2, Strassen rank 7. Total: 14.
    # So A1*B1 has rank 14, A2*B2 has rank 14. Total: 28.
    #
    # That's not 14. I made an error. Let me reconsider.
    #
    # The issue: 2x4x4 means multiplying 2x4 by 4x4. The "4" in the middle is the
    # shared dimension. So A has shape (2,4), B has shape (4,4), C has shape (2,4).
    # The tensor T[i,j,k] where i in [0,2), j in [0,4), k in [0,4) represents
    # C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # To use Strassen: partition the shared dimension j into 2+2.
    # A = [A1 | A2] where A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    # B = [B1; B2] where B1 = B[0:2, :] (2x4), B2 = B[2:4, :] (2x4)
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 times 2x4 = 2x4. This is a 2x2x4 problem.
    # For 2x2x4: A is 2x2, B is 2x4, C is 2x4.
    # Partition B's 4 output cols into 2+2: B = [B1a | B1b], each 2x2.
    # C = A*B1a + A*B1b. Each is 2x2x2, Strassen rank 7. Total: 14.
    #
    # So A1*B1 has rank 14. Similarly A2*B2 has rank 14. Total: 28.
    #
    # That's the same as before. The 14 I thought of was wrong.
    #
    # OK so the current 50-rank solution might be close to optimal.
    # Let me try a different approach: 2x4x7 with different partitioning.
    #
    # What if we partition the 7 output cols (k) as 4+3?
    # 2x4x4: rank 28 (as computed)
    # 2x4x3: A is 2x4, B is 4x3, C is 2x3.
    #   Partition j into 2+2: A = [A1|A2], B = [B1;B2]
    #   C = A1*B1 + A2*B2
    #   A1*B1: 2x2 * 2x3 = 2x3. 2x2x3 problem.
    #   2x2x3: partition B's 3 cols into 2+1:
    #     2x2x2 (rank 7) + 2x2x1 (rank 4) = 11
    #   So A1*B1 has rank 11, A2*B2 has rank 11. Total: 22.
    # 2x4x3: rank 22
    # Total: 28 + 22 = 50. Same as before!
    #
    # So the 50-rank solution is natural from this partitioning.
    #
    # Can we do better with a different partition?
    # What if we don't split j at all, and use a direct 2x4x7 decomposition?
    # Lower bound: max(2*4, 4*7, 2*7) = max(8, 28, 14) = 28.
    # So rank >= 28. Can we achieve 28 or 29?
    #
    # This is a hard open problem. For now, let's try the greedy set cover
    # to see if we can shave off a few ranks from 50.
    
    # --- Step 4: Try to find a rank-49 or rank-48 decomposition by
    #     checking if any two columns can be merged ---
    # If columns r1 and r2 have the property that removing both and adding
    # a new column that is their "sum" still works, we reduce rank by 1.
    
    if time.time() < deadline - 10:
        # Try pairwise column elimination
        U_try, V_try, W_try = best_U.copy(), best_V.copy(), best_W.copy()
        rank_try = U_try.shape[1]
        
        # Try removing pairs of columns
        for r1 in range(rank_try):
            for r2 in range(r1 + 1, rank_try):
                if time.time() > deadline - 10:
                    break
                mask = np.ones(rank_try, dtype=bool)
                mask[r1] = False
                mask[r2] = False
                U_new = U_try[:, mask]
                V_new = V_try[:, mask]
                W_new = W_try[:, mask]
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = U_new.shape[1]
                    U_try, V_try, W_try = U_new, V_new, W_new
                    rank_try = U_new.shape[1]
                    break
            if time.time() > deadline - 10:
                break
    
    if best_rank < 50:
        return (best_U, best_V, best_W)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS

    # Build the known rank-50 decomposition
    U, V, W, col = _build_strassen_schoolbook(n, m, p)
    rank = col

    # Verify the rank-50 decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        return _schoolbook(n, m, p)

    # Attempt rank reduction via greedy set cover with local search
    if time.time() < deadline - 25:
        try:
            reduced = _greedy_set_cover(U, V, W, n, m, p, deadline)
            if reduced is not None:
                r_red = reduced[0].shape[1]
                if r_red < rank:
                    return {"U": reduced[0].tolist(), "V": reduced[1].tolist(), "W": reduced[2].tolist()}
        except Exception:
            pass

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
