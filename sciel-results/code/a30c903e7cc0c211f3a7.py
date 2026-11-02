"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] where C[k,i] = sum_j A[i,j]*B[j,k].
    T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the tensor."""
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    # Reconstruct: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    rank = U.shape[1]
    # Efficient: for each r, add outer product contribution
    recon = np.zeros_like(T, dtype=np.int64)
    for r in range(rank):
        ur = U[:, r]
        vr = V[:, r]
        wr = W[:, r]
        # outer product of three vectors
        recon += np.einsum('a,b,c->abc', ur, vr, wr)
    return np.array_equal(recon, T)

def _greedy_rank_reduction(n, m, p, max_rank=50, deadline=None):
    """Try to find a lower-rank decomposition by starting from known blocks
    and greedily attempting to merge columns."""
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 130

    # Start with the block decomposition (rank 50)
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

    if col != rank:
        return None

    # Now try to reduce rank by finding linear dependencies among columns
    # Strategy: try removing one column at a time and see if the remaining
    # columns can be adjusted (scaled/shifted) to still decompose the tensor.
    # This is expensive, so use a smarter approach:
    
    # Alternative structural approach: try different block decompositions
    # 1) Try using 2x2x2 Strassen for all 6 pairs of (j,k) with n=2
    #    But we only have 2 j-pairs and 3.5 k-pairs (0-1, 2-3, 4-5, 6)
    #    For the last k=6, we need a different treatment.
    
    # Try: use Strassen for j-pairs (0,1) and (2,3), and for k:
    #   k-pairs (0,1), (2,3), (4,5) -> 3 Strassen blocks per j-pair = 6 blocks * 7 = 42
    #   k=6: need 2x2x1 for each j-pair -> 2 blocks * 4 = 8
    #   Total = 50 (same as before)
    
    # Try alternative: Can we combine the k=6 schoolbook with adjacent Strassen?
    # Or use a different decomposition for the 2x4x7 case?
    
    # Structural idea: Use 2x2x2 Strassen for (j,k) pairs but handle the
    # irregular k=6 differently. What if we group k as (0,1,2,3,4,5,6)
    # and use a 2x2x3 decomposition for k=4,5,6?
    
    # 2x2x3 matrix multiplication: C[2x3] = A[2x2] * B[2x3]
    # Standard: 2*2*3 = 12 multiplications
    # Can we do better? 2x2x3 is equivalent to two 2x2x1 problems.
    # Rank of 2x2x3 tensor: known to be 8 (two 2x2x1 blocks, each rank 4)
    # Actually 2x2x1 has rank 4 (schoolbook), so 2x2x3 = 8.
    # But wait, can we share? The A matrix is shared between the two k-columns.
    
    # Let's try: for k=4,5,6 with a 2x2 A block:
    # C[:,4] = A * B[:,4], C[:,5] = A * B[:,5], C[:,6] = A * B[:,6]
    # This is a 2x2x3 multiplication.
    # We can decompose as: use Strassen-like for the first two k's,
    # and add one more for the third.
    
    # Actually, let's try a completely different structural approach:
    # Decompose 2x4x7 as:
    # - Split j into two 2x2 blocks: j=(0,1) and j=(2,3)
    # - For each j-block, we have 2x2x7 multiplication
    # - 2x2x7: can we do better than 7*4=28? 
    #   2x2x2 Strassen (7 mults) for k=0,1 + 2x2x2 Strassen for k=2,3 + 2x2x3 for k=4,5,6
    #   2x2x3 rank: let's compute. C[2x3] = A[2x2]*B[2x3]
    #   = A * [B[:,0] B[:,1] B[:,2]]
    #   Each column is a 2x2x1 problem (rank 4), but A is shared.
    #   Total rank for 2x2x3: we can use 4+4+4=12, or can we share?
    #   Actually no sharing possible since each output column is independent.
    #   So 2x2x3 rank = 12? No wait, 2x2x1 has rank 4, so 3 columns = 12.
    #   Hmm, that's worse than just doing 2x2x2 + 2x2x1 = 7+4 = 11.
    #   
    #   So 2x2x7 = 7 (Strassen k0-1) + 7 (Strassen k2-3) + 7 (Strassen k4-5) + 4 (schoolbook k6) = 25
    #   Two j-blocks: 2*25 = 50. Same as before.
    
    # What if we use a different split? Split j into (0,1,2) and (3)?
    # 2x3x7 + 2x1x7
    # 2x1x7: rank 7 (each output needs one mult per k)
    # 2x3x7: 2*3*7 = 42 schoolbook, or can we do better?
    #   2x3x7 = for each of 7 k's, 2x3x1 (rank 6) = 42. No improvement.
    
    # What about splitting k differently?
    # 2x4x7: split k into (0,1,2,3,4,5) and (6)
    # 2x4x6 + 2x4x1
    # 2x4x1: rank 8 (each of 8 products)
    # 2x4x6: 2*4*6 = 48 schoolbook
    #   Split j: (0,1) and (2,3), each 2x2x6
    #   2x2x6: Strassen for k0-1 (7) + Strassen for k2-3 (7) + Strassen for k4-5 (7) = 21
    #   Two j-blocks: 42
    #   Total: 42 + 8 = 50. Same!
    
    # The rank 50 seems to be the natural block decomposition limit.
    # To go lower, we need cross-block sharing or a fundamentally different approach.
    
    # Try: can any two Strassen blocks share a factor?
    # Strassen block for (j=0,1; k=0,1) uses A entries: A[0,0],A[0,1],A[1,0],A[1,1]
    # Strassen block for (j=2,3; k=0,1) uses A entries: A[0,2],A[0,3],A[1,2],A[1,3]
    # No overlap in A entries, so no sharing possible.
    
    # Within a j-block, different k-blocks:
    # (j=0,1; k=0,1) uses B: B[0,0],B[0,1],B[1,0],B[1,1]
    # (j=0,1; k=2,3) uses B: B[0,2],B[0,3],B[1,2],B[1,3]
    # No overlap in B entries either.
    
    # So the block decomposition is tight at 50 for this partitioning.
    
    # New structural idea: What if we DON'T use 2x2 Strassen blocks?
    # Instead, use a different algebraic decomposition.
    # For 2x2x2, Strassen is optimal (rank 7).
    # For 2x2x1, rank is 4 (schoolbook is optimal).
    # 
    # What about a 2x4x2 block? C[2x2] = A[2x4] * B[4x2]
    # Rank: 2*4*2 = 16 schoolbook.
    # Can we do better? This is equivalent to computing two 2x4x1 products.
    # Each 2x4x1 has rank 8. Total 16. No improvement with separate.
    # But with sharing: A is shared between the two output columns.
    # C[:,0] = A*B[:,0], C[:,1] = A*B[:,1]
    # We can write: C = A * B where B is 4x2.
    # This is a 2x4x2 tensor. Known rank? 
    # For 2xnx2, the rank is known. 2x2x2=7, 2x3x2=? 2x4x2=?
    # 2x4x2: I believe rank is 12 (not 16).
    # Let's verify: C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1; j=0,1,2,3; k=0,1
    # We can use: 
    #   M1 = A[0,0]*B[0,0] -> contributes to C[0,0]
    #   ... this gets complex.
    # 
    # Actually, 2xnx2 matrix multiplication:
    # C[0,0] = A00*B00 + A01*B10 + A02*B20 + A03*B30
    # C[0,1] = A00*B01 + A01*B11 + A02*B21 + A03*B31
    # C[1,0] = A10*B00 + A11*B10 + A12*B20 + A13*B30
    # C[1,1] = A10*B01 + A11*B11 + A12*B21 + A13*B31
    #
    # Strassen-like for 2x4x2:
    # Group A columns: A_left = [A00,A10], A_mid = [A01,A11], A_right = [A02,A12], A_far = [A03,A13]
    # Group B rows: B_top = [B00,B01], B_next = [B10,B11], B_next2 = [B20,B21], B_bot = [B30,B31]
    #
    # M1 = (A00+A10)*(B00+B10+B20+B30) -- no, this doesn't work simply.
    #
    # Let me try a different approach. For 2x4x2:
    # We can write it as 2x2x2 + 2x2x2 by splitting j:
    # C = A_left * B_top2 + A_right * B_bot2 where B_top2 = B[0:2,:], B_bot2 = B[2:4,:]
    # Each 2x2x2 has rank 7, total 14.
    # Or split differently: 2x4x2 = (2x2x2) + (2x2x2) = 14.
    # Better than 16!
    #
    # So for 2x4x7:
    # Split k into pairs: (0,1), (2,3), (4,5), and single 6.
    # For each k-pair: 2x4x2 with rank 14. Three pairs: 42.
    # For k=6: 2x4x1 with rank 8.
    # Total: 42 + 8 = 50. Same again!
    #
    # But wait, can we do 2x4x2 with rank less than 14?
    # 2x4x2 = A[2x4] * B[4x2]. 
    # Split A into two 2x2 blocks: A = [A1 | A2] where A1,A2 are 2x2.
    # B = [B1; B2] where B1,B2 are 2x2.
    # C = A1*B1 + A2*B2.
    # Each product is 2x2x2 (rank 7), sum has rank at most 14.
    # Can we share? The outputs are summed, so we need to add the results.
    # In tensor decomposition, C = A1*B1 + A2*B2 means:
    # C[i,k] = sum_j1 A1[i,j1]*B1[j1,k] + sum_j2 A2[i,j2]*B2[j2,k]
    # This is just the concatenation: treat as 2x4x2 directly.
    # The rank of 2x4x2 tensor: I'll compute it.
    # Actually, I recall that for 2xnx2, the rank is 2n for n>=2 when using schoolbook,
    # but Strassen can reduce it. For 2x2x2 it's 7 < 8=2*2*2/2... 
    # Hmm, let me just try rank 12 for 2x4x2.
    #
    # 2x4x2 decomposition with rank 12:
    # C[0,0] = A00*B00 + A01*B10 + A02*B20 + A03*B30
    # C[0,1] = A00*B01 + A01*B11 + A02*B21 + A03*B31
    # C[1,0] = A10*B00 + A11*B10 + A12*B20 + A13*B30
    # C[1,1] = A10*B01 + A11*B11 + A12*B21 + A13*B31
    #
    # Try:
    # M1 = (A00+A10) * (B00+B10+B20+B30) -- 1 mult, contributes to all 4 outputs
    # This is getting complicated. Let me just verify rank 50 is correct
    # and try a search-based approach to see if we can find rank < 50.
    
    # For now, return the rank-50 solution as baseline.
    # Future: implement actual search.
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_reduce_rank(n, m, p, initial_U, initial_V, initial_W, deadline=None):
    """Try to reduce rank by removing columns and checking validity."""
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 125
    
    U = np.array(initial_U, dtype=np.int64)
    V = np.array(initial_V, dtype=np.int64)
    W = np.array(initial_W, dtype=np.int64)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each column one at a time
    for trial in range(rank):
        if time.time() > deadline:
            break
        # Try removing column 'trial'
        mask = [i for i in range(rank) if i != trial]
        U_trial = U[:, mask]
        V_trial = V[:, mask]
        W_trial = W[:, mask]
        
        # Check if this still decomposes the tensor
        # This is expensive, so only do it for a few trials
        if trial < 5:  # Limit expensive checks
            if _verify(U_trial.tolist(), V_trial.tolist(), W_trial.tolist()):
                best_rank = rank - 1
                best_U, best_V, best_W = U_trial, V_trial, W_trial
                U, V, W = U_trial, V_trial, W_trial
                rank = rank - 1
                # Reset trial counter since rank changed
                trial = -1
    
    if best_rank < rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # First, build the rank-50 block decomposition
    result = _greedy_rank_reduction(n, m, p, deadline=deadline)
    
    if result is None:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
    
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    # Try to reduce rank
    if time.time() < deadline - 10:
        reduced = _try_reduce_rank(n, m, p, U, V, W, deadline=deadline)
        if reduced is not None:
            U, V, W, new_rank = reduced
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

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
