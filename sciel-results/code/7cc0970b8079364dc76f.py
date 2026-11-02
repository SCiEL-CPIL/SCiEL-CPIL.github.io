"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = 2, 2, 13
    
    # Structural change: Use a global Strassen-style approach across all 13 columns
    # instead of block-by-block. The key insight is that the A-matrix factors in
    # Strassen's algorithm are the same regardless of which B-columns they multiply.
    # 
    # For 2x2 * 2x2 Strassen:
    # P1 = (a00+a11) * (b00+b11)
    # P2 = (a10+a11) * b00
    # P3 = a00 * (b10-b11)
    # P4 = a11 * (b01-b00)
    # P5 = (a00+a01) * b11
    # P6 = (a10-a00) * (b00+b10)
    # P7 = (a01-a11) * (b01+b11)
    #
    # For 13 columns, we can define linear combinations of B columns that serve
    # as the "b" entries in each Strassen product. Since the A-side factors are
    # shared, we get 7 products total, each involving a linear combination of all
    # 13 columns of B.
    #
    # But wait - each Strassen product produces a 2x2 result (4 scalar products).
    # With 13 columns, we need 26 output scalars (2 rows x 13 columns).
    # 7 Strassen products give us 7*4 = 28 scalar outputs, which is >= 26.
    # However, the 7 products are linearly dependent in general, so this doesn't
    # directly work as a rank-7 decomposition.
    #
    # Let me reconsider. The matrix multiplication tensor for 2x2 * 2x13 has
    # dimensions 4 x 26 x 26. The rank of this tensor is at least max(4, 26, 26) = 26.
    # 
    # A better structural approach: Use the fact that A is 2x2 and B is 2x13.
    # We can write:
    #   C = A * B
    # where C is 2x13.
    #
    # Let's use a rank-r decomposition where r < 46.
    # 
    # Key idea: Since A has only 4 entries, and we need to multiply each entry
    # of A with each entry of B (26 entries), the naive approach needs 4*26 = 104
    # products, but tensor rank is much less.
    #
    # Let me try a different structural approach: decompose using the fact that
    # we can group B columns and use Strassen within groups, but also share
    # computations across groups.
    #
    # Actually, let me try to find a decomposition with rank 39 or less by
    # using a smarter grouping.
    #
    # New approach: Treat the 13 columns as 13 separate "blocks" and find a
    # decomposition that shares A-side factors. 
    #
    # For each column j of B, we compute A * B[:,j] which is a 2x1 vector.
    # This is a 2x2 * 2x1 multiplication, which has tensor rank 4 (schoolbook).
    # But we can do better: 2x2 * 2x1 has rank 4 (it's just 4 scalar products).
    #
    # Wait, 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1.
    # C[0] = a00*b0 + a01*b1
    # C[1] = a10*b0 + a11*b1
    # This is 4 scalar products, so rank 4.
    #
    # For 13 columns: 13 * 4 = 52 (schoolbook). Strassen blocks give 46.
    #
    # Let me try yet another approach: Can we use rank 45 or less?
    # 
    # Structural change: Use a direct algebraic construction with rank 39.
    # 
    # The idea: Since A is 2x2, we can express the multiplication as:
    #   C = A @ B
    # 
    # Let's define new variables to reduce the number of products.
    # 
    # Actually, let me try a completely different structural approach:
    # Use the Winograd minimal filtering algorithm idea. For 2x2 * 2x13,
    # we can think of this as a convolution-like operation and apply
    # Winograd's method.
    #
    # Winograd for 2x2 * 2x2 uses 4 products (rank 4) instead of 7 (Strassen).
    # Wait no, Winograd for 2x2 matrix multiply is rank 4? Let me check.
    # 
    # For 2x2 * 2x2:
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    #
    # Winograd: 
    # m0 = a00 + a01, m1 = a01, m2 = a10 + a11, m3 = a11
    # m4 = b00 + b01, m5 = b01, m6 = b10 + b11, m7 = b11
    # p0 = m0*m4, p1 = m1*m5, p2 = m2*m6, p3 = m3*m7
    # C00 = p0 - p1 - p2 + p3... no this doesn't work simply.
    #
    # Actually the minimal rank for 2x2*2x2 is 7 (Strassen is optimal).
    # 
    # Let me try a completely different structural approach:
    # 
    # Use rank 39 by exploiting the structure more carefully.
    # 
    # For each pair of B columns (2j, 2j+1), Strassen gives rank 7.
    # For the last single column, schoolbook gives rank 4.
    # Total: 6*7 + 4 = 46.
    #
    # Can we do the last column with fewer than 4? No, 2x2*2x1 needs rank 4.
    #
    # Can we do a pair of columns with fewer than 7? 
    # 2x2 * 2x2 has minimal rank 7, so no.
    #
    # What if we group columns differently? E.g., 3 columns at a time?
    # 2x2 * 2x3: what's the minimal rank?
    # The tensor has dimensions 4 x 6 x 6. 
    # Lower bound: max(4, 6, 6) = 6.
    # Upper bound: 3*7/2... no, let me think.
    # 
    # 2x2 * 2x3 can be seen as three 2x2*2x1 multiplications, each rank 4, total 12.
    # Or as 2x2*2x2 + 2x2*2x1 = 7 + 4 = 11.
    # Can we do better? 
    #
    # Actually, let me just try to verify if rank 46 is achievable and correct,
    # and if there's a way to get to 45 or 44.
    #
    # Let me try: 6 pairs (rank 7 each) + 1 single (rank 4) = 46.
    # Alternative: 5 pairs (rank 7 each) + 3 singles (rank 4 each) = 35 + 12 = 47. Worse.
    # Alternative: 6 pairs + 1 single = 46.
    # Alternative: 4 pairs + 5 singles = 28 + 20 = 48. Worse.
    #
    # So 46 seems to be the best with this block approach.
    #
    # Let me try to see if I can reduce the single column cost.
    # For a single column j, we need:
    # C[0,j] = a00*B[0,j] + a01*B[1,j]
    # C[1,j] = a10*B[0,j] + a11*B[1,j]
    # This is 4 products, so rank 4. Can't reduce.
    #
    # What if I use a different decomposition for the whole thing?
    #
    # New structural idea: Use the fact that the A matrix is 2x2 and apply
    # a global decomposition. Let me try to find a rank-39 or rank-38 decomposition.
    #
    # Actually, let me just carefully implement the rank-46 decomposition and
    # verify it's correct. The previous code had issues with the Strassen factors.

    # Strassen decomposition for 2x2 * 2x2:
    # A indices: a00=0, a01=1, a10=2, a11=3
    # B indices: b00=0, b01=1, b10=2, b11=3
    # C indices: c00=0, c01=1, c10=2, c11=3
    
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b10)
    # P7 = (a01-a11)(b01+b11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P1 - P2 + P6
    # C10 = P2 + P4 - P3 + P5
    # C11 = P1 - P4 + P2 - P6
    
    # U (A-side, 4x7):
    # col t: P1  P2  P3  P4  P5  P6  P7
    # a00:    1   0   1   0   1  -1   0
    # a01:    0   0   0   0   1   0   1
    # a10:    0   1   0   0   0   1   0
    # a11:    1   1   0   1   0   0  -1
    
    # V (B-side, 4x7):
    # b00:    1   1   0   0   0   1   0
    # b01:    0   0   0   1   0   0   1
    # b10:    0   0   1  -1   0   1   0
    # b11:    1   0  -1   0   1   0   1
    
    # W (C-side, 4x7):
    # c00:    1   0   0   1   0   0   1
    # c01:    1  -1   1   0   0   1   0
    # c10:    0   1   1   0  -1   0   0
    # c11:    1   1   0  -1   0  -1   0
    
    strassen_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=float)
    
    strassen_W = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [1, -1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, -1, 0, 0],
        [1, 1, 0, -1, 0, -1, 0],
    ], dtype=float)
    
    # Verify Strassen decomposition
    S_target = np.zeros((4, 4, 4), dtype=float)
    S_target[0, 0, 0] = 1  # a00*b00 -> C00
    S_target[0, 2, 2] = 1  # a00*b10 -> C10
    S_target[1, 1, 1] = 1  # a01*b01 -> C01
    S_target[1, 3, 3] = 1  # a01*b11 -> C11
    S_target[2, 0, 1] = 1  # a10*b00 -> C01
    S_target[2, 1, 3] = 1  # a10*b01 -> C11
    S_target[3, 2, 1] = 1  # a11*b10 -> C01
    S_target[3, 3, 3] = 1  # a11*b11 -> C11
    
    S_recon = np.einsum('at,bt,ct->abc', strassen_U, strassen_V, strassen_W)
    
    if not np.allclose(S_recon, S_target, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][column] = 1
                    V[k * p + j][column] = 1
                    W[j * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build full decomposition with rank 46
    total_rank = 6 * 7 + 4  # = 46
    
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for block in range(6):
        j0 = 2 * block
        j1 = 2 * block + 1
        for t in range(7):
            U_full[:, col] = strassen_U[:, t]
            for k in range(2):
                for jl in range(2):
                    j_global = j0 + jl
                    v_local = k * 2 + jl
                    v_global = k * p + j_global
                    V_full[v_global, col] = strassen_V[v_local, t]
            for jl in range(2):
                for i in range(2):
                    j_global = j0 + jl
                    w_local = jl * 2 + i
                    w_global = j_global * n + i
                    W_full[w_global, col] = strassen_W[w_local, t]
            col += 1
    
    # Schoolbook for column 12
    j = 12
    for i in range(n):
        for k in range(m):
            U_full[i * m + k, col] = 1
            V_full[k * p + j, col] = 1
            W_full[j * n + i, col] = 1
            col += 1
    
    # Verify full decomposition
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_target[i * m + k, k * p + j, j * n + i] = 1
    
    T_recon = np.einsum('at,bt,ct->abc', U_full, V_full, W_full)
    
    if not np.allclose(T_recon, T_target, atol=1e-10):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][column] = 1
                    V[k * p + j][column] = 1
                    W[j * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies among rank terms
    # A rank term t is redundant if there exist coefficients c_s (s != t) such that
    # U[:,t] = sum_s c_s U[:,s], V[:,t] = sum_s c_s V[:,s], W[:,t] = sum_s c_s W[:,s]
    # This is a strong condition. Instead, try greedy elimination:
    # For each term t, check if removing it and adjusting others still gives exact decomposition.
    
    # Structural change: Try a different block decomposition
    # Use 4 Strassen blocks (8 cols, rank 28) + 5 schoolbook columns (rank 20) = 48
    # vs current 6 Strassen + 1 schoolbook = 46. Current is better.
    
    # Try: Can we do 2x2 x 2x3 with rank 8?
    # Let's try a custom decomposition for 3 columns.
    # Actually, let's try a completely different structural approach:
    # Use rank-42 by finding a 2x2 x 2x3 decomposition with rank 7.
    
    # For 2x2 x 2x3, try:
    # p1 = (a00+a11)(b00+b10+b01+b11+b02+b12)  -- no, this mixes rows
    # 
    # Let me try a systematic approach: use 5 Strassen pairs + 1 three-column block
    # For the 3-column block, use schoolbook: rank 4*3 = 12. Total: 5*7+12 = 47. Worse.
    # 
    # Alternative: 4 Strassen pairs + 2 three-column blocks: 4*7 + 2*12 = 44. Better!
    # But wait, 4 pairs cover 8 columns, 2 three-column blocks cover 6 columns, total 14 > 13.
    # 4 pairs (8 cols) + 1 three-col (3 cols) + 1 single (1 col) = 12 cols. Missing 1.
    # 4 pairs (8) + 2 three-col (6) = 14. Too many.
    # 3 pairs (6) + 2 three-col (6) + 1 single (1) = 13. Rank: 3*7 + 2*12 + 4 = 21+24+4 = 49. Worse.
    #
    # So the 6-pair + 1-single = 46 is still best with this approach.
    #
    # Let me try a truly different structural family: use a "global" decomposition
    # where I define 42 rank terms with a different algebraic structure.
    #
    # Idea: Use the fact that C = A*B can be written as:
    # C[0,:] = a00*B[0,:] + a01*B[1,:]
    # C[1,:] = a10*B[0,:] + a11*B[1,:]
    #
    # Define: D0 = a00*B[0,:] + a01*B[1,:]  (this is C[0,:])
    #         D1 = a10*B[0,:] + a11*B[1,:]  (this is C[1,:])
    #
    # Each Di is a 1x13 vector computed from 2 A-entries and 26 B-entries.
    # Computing Di requires 2*13 = 26 scalar products (schoolbook).
    # Total: 52. No improvement.
    #
    # But if we can share products between D0 and D1...
    # D0[j] = a00*B[0,j] + a01*B[1,j]
    # D1[j] = a10*B[0,j] + a11*B[1,j]
    #
    # For a fixed j, these 4 products (a00*B[0,j], a01*B[1,j], a10*B[0,j], a11*B[1,j])
    # are all needed. No sharing possible within a column.
    #
    # Across columns: a00*B[0,j] for different j are independent products.
    # Unless we combine: a00*(B[0,0]+B[0,1]) as one product, but then we can't
    # separate them in the output.
    #
    # UNLESS we use the W matrix creatively. If we have:
    # p = a00 * (B[0,0] + B[0,1])
    # Then C[0,0] gets W[0,t]*p and C[0,1] gets W[1,t]*p.
    # We need C[0,0] to get a00*B[0,0] and C[0,1] to get a00*B[0,1].
    # With W[0,t]=1 and W[1,t]=1, both get a00*(B[0,0]+B[0,1]). Wrong.
    #
    # So no sharing across columns is possible with this simple approach.
    #
    # The fundamental limit: the tensor rank of M_{2,2} x M_{2,13} is at least
    # the border rank, which is at least 26 (dimension of output space).
    # Known upper bound: 46 (Strassen blocks).
    #
    # Let me try to verify if rank 45 is possible by checking if one term
    # can be eliminated from the rank-46 decomposition.
    
    # Check if any single term can be removed
    for t_remove in range(total_rank):
        if time.time() > deadline - 30:
            break
        U_test = np.delete(U_full, t_remove, axis=1)
        V_test = np.delete(V_full, t_remove, axis=1)
        W_test = np.delete(W_full, t_remove, axis=1)
        T_test = np.einsum('at,bt,ct->abc', U_test, V_test, W_test)
        if np.allclose(T_test, T_target, atol=1e-10):
            # Found a redundant term!
            U_full = U_test
            V_full = V_test
            W_full = W_test
            total_rank -= 1
            break
    
    # Try removing pairs of terms
    for t1 in range(total_rank):
        for t2 in range(t1+1, total_rank):
            if time.time() > deadline - 30:
                break
            mask = [i for i in range(total_rank) if i != t1 and i != t2]
            U_test = U_full[:, mask]
            V_test = V_full[:, mask]
            W_test = W_full[:, mask]
            T_test = np.einsum('at,bt,ct->abc', U_test, V_test, W_test)
            if np.allclose(T_test, T_target, atol=1e-10):
                U_full = U_test
                V_full = V_test
                W_full = W_test
                total_rank -= 2
                break
        if time.time() > deadline - 30:
            break
    
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
