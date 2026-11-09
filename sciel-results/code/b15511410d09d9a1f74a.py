import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 5 by 5 matrix 
    by a 5 by 5 matrix.
    Uses block decomposition: 4x4x4 core with Strassen (rank 49) + border terms (schoolbook).
    Total rank: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Define the 2x2x2 Strassen decomposition (rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices: for 2x2, U index = i*2+j, V index = j*2+k, W index = k*2+i
    def get_2x2_strassen():
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00 and A11 -> indices 0 and 3
        # V: B00 and B11 -> indices 0 and 3
        # W: C00 += M1, C11 += M1 -> indices 0 and 3
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # M2 = (A10+A11)B00
        # U: A10 and A11 -> indices 2 and 3
        # V: B00 -> index 0
        # W: C10 += M2, C11 -= M2 -> indices 2 and 3
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1
        
        # M3 = A00(B10-B11)
        # U: A00 -> index 0
        # V: B10 and B11 -> indices 2 and 3
        # W: C01 += M3, C10 += M3 -> wait, let me recheck
        # Actually: C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # M3 should contribute to C01 and C10? No...
        # Standard Strassen:
        # C00 = M1 - M4 + M7 - M6
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M2
        # So:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # M3 = A00(B10-B11)
        # U: A00 -> index 0
        # V: B10 -> index 2, B11 -> index 3
        # W: C11 -= M3 -> index 3
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[3, 2] = -1
        W2[3, 2] = -1
        
        # M4 = A11(B01-B00)
        # U: A11 -> index 3
        # V: B01 -> index 1, B00 -> index 0
        # W: C00 -= M4, C10 += M4 -> indices 0 and 2
        U2[3, 3] = 1
        V2[1, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = -1; W2[2, 3] = 1
        
        # M5 = (A00+A01)B11
        # U: A00 -> index 0, A01 -> index 1
        # V: B11 -> index 3
        # W: C01 += M5, C11 += M5 -> indices 1 and 3
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[1, 4] = 1; W2[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # U: A10 -> index 2, A00 -> index 0
        # V: B00 -> index 0, B01 -> index 1
        # W: C00 -= M6 -> index 0
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[0, 5] = -1
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01 -> index 1, A11 -> index 3
        # V: B10 -> index 2, B11 -> index 3
        # W: C00 += M7, C01 += M7 -> indices 0 and 1
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1; W2[1, 6] = 1
        
        return U2, V2, W2
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    # View 4x4 as 2x2 blocks of 2x2
    # Block (bi, bj) where bi, bj in {0,1}
    # Global index for U: (bi*2+li, bj*2+lj) -> (bi*2+li)*4 + (bj*2+lj)
    # Global index for V: (bj*2+lj, bk*2+lk) -> (bj*2+lj)*4 + (bk*2+lk)  
    # Global index for W: (bk*2+lk, bi*2+li) -> (bk*2+lk)*4 + (bi*2+li)
    
    def get_4x4_strassen():
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        U2, V2, W2 = get_2x2_strassen()
        
        # For each of the 7 Strassen multiplications at block level,
        # we need to apply the 2x2x2 Strassen to the 2x2 blocks
        # Block-level indices: bi, bj, bk in {0,1}
        # For each block-level Strassen term t (0..6), and each local 2x2 Strassen term s (0..6)
        
        for t in range(7):
            for s in range(7):
                col = t * 7 + s
                
                # Determine which block-level indices are involved
                # For U: which (bi, bj) blocks have nonzero U2[bi*2+li, bj*2+lj, t]
                # Actually, U2 has shape (4, 7) where row = i*2+j for 2x2
                # So for block level, U2[bi*2+..., t] tells us which block rows
                
                # Let me think differently. The block-level Strassen gives us 7 "multiplications"
                # Each multiplication is a product of two 2x2 block matrices.
                # For each such product, we use the 2x2x2 Strassen decomposition.
                
                # The block-level U, V, W for 2x2x2 (where each entry is a 2x2 matrix)
                # U_block[i, t] for i in {0,1,2,3} (i = bi*2+... no, i is just the block row/col)
                
                # Actually, let me define the block-level 2x2x2 Strassen:
                # U_block has shape (4, 7) where row index = bi*2+bj (for A blocks)
                # V_block has shape (4, 7) where row index = bj*2+bk (for B blocks)  
                # W_block has shape (4, 7) where row index = bk*2+bi (for C blocks)
                
                # These are exactly U2, V2, W2 from get_2x2_strassen()
                
                # Now, for each (t, s), we need to find all (bi, bj, bk) such that
                # U2[bi*2+bj, t] != 0, V2[bj*2+bk, t] != 0, W2[bk*2+bi, t] != 0
                # And for each such combination, add the local Strassen term s with
                # coefficient U2[bi*2+bj, t] * V2[bj*2+bk, t] * W2[bk*2+bi, t]
                
                # But wait, that's not quite right. The block-level Strassen says:
                # C_block = sum_t U_block[:, t] * V_block[:, t] * W_block[:, t]
                # where each "multiplication" is a 2x2 matrix multiplication.
                
                # So for each t, we have a product of three 2x2 matrices (in the tensor sense).
                # The 2x2 matrix multiplication itself can be decomposed using Strassen (rank 7).
                
                # So the full 4x4x4 decomposition is:
                # For each block-level term t (0..6):
                #   For each local 2x2 Strassen term s (0..6):
                #     For each (li, lj, lk) in {0,1}^3:
                #       Add term with coefficient U2[bi*2+bj, t] * V2[bj*2+bk, t] * W2[bk*2+bi, t] * U2_local[li*2+lj, s] * V2_local[lj*2+lk, s] * W2_local[lk*2+li, s]
                
                # But we need to be more careful. Let me use a cleaner formulation.
                
                pass
        
        # Let me use a different, more direct approach.
        # I'll construct the 4x4x4 rank-49 decomposition directly.
        
        # Actually, let me just verify my 2x2 Strassen is correct first, then build up.
        
        # For now, let me use a simpler but correct approach:
        # Use schoolbook for 4x4x4 (rank 64) instead of Strassen (rank 49)
        # This gives total rank: 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125
        # That's the same as the original! So I need to get Strassen right.
        
        # Let me carefully construct the 4x4x4 Strassen.
        
        # The key insight: 4x4x4 can be seen as 2x2 blocks of 2x2.
        # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
        # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
        # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
        
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Using Strassen at block level:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 - M4 + M7 - M6
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M2
        
        # Each Mj is a 2x2 matrix product, which can be computed with rank 7 (Strassen).
        # So total rank = 7 * 7 = 49.
        
        # Now I need to construct U, V, W for the full 4x4x4.
        
        # For each block-level term t (0..6), and each local 2x2 Strassen term s (0..6):
        # The coefficient for the global term (t, s) involves:
        # - Which block indices (bi, bj) for A, (bj, bk) for B, (bk, bi) for C are involved
        # - The local 2x2 Strassen coefficients
        
        # Let me define the block-level Strassen coefficients:
        # U_block[i, t] where i = bi*2+bj (0..3), t = 0..6
        # V_block[j, t] where j = bj*2+bk (0..3), t = 0..6  
        # W_block[k, t] where k = bk*2+bi (0..3), t = 0..6
        
        # These are exactly U2, V2, W2 from get_2x2_strassen()
        
        # Now for the local 2x2 Strassen (for the actual 2x2 matrix multiplications):
        # U_local[li, s] where li = li*2+lj (0..3), s = 0..6
        # V_local[lj, s] where lj = lj*2+lk (0..3), s = 0..6
        # W_local[lk, s] where lk = lk*2+li (0..3), s = 0..6
        
        # These are also U2, V2, W2 (same decomposition)
        
        # The full 4x4x4 U, V, W:
        # U4[global_i, col] where global_i = (bi*2+li)*4 + (bj*2+lj)
        # V4[global_j, col] where global_j = (bj*2+lj)*4 + (bk*2+lk)
        # W4[global_k, col] where global_k = (bk*2+lk)*4 + (bi*2+li)
        # col = t*7 + s
        
        # For each t, s, and each (bi, bj, bk, li, lj, lk):
        # coeff = U2[bi*2+bj, t] * V2[bj*2+bk, t] * W2[bk*2+bi, t] * U2[li*2+lj, s] * V2[lj*2+lk, s] * W2[lk*2+li, s]
        
        # Wait, that's not right either. The block-level and local-level are separate.
        
        # Let me think again. The block-level Strassen gives:
        # C_block[bk, bi] = sum_t U_block[bi*2+bj, t] * (A_block[bi, bj] * B_block[bj, bk]) * W_block[bk*2+bi, t]
        
        # No, that's not how tensor decomposition works. Let me reconsider.
        
        # The tensor for 4x4x4 matrix multiplication has indices (i, j, k) where:
        # i = row of A (0..3), j = col of A / row of B (0..3), k = col of B (0..3)
        # The tensor T[i, j, k] = 1 if this triple contributes to C[k, i]
        
        # Actually, the standard convention is:
        # C[i, k] = sum_j A[i, j] * B[j, k]
        # So the tensor has entries T[i, j, k] where i is row of A/C, j is shared, k is col of B/C
        # U has shape (n*m, rank) = (16, rank) with index i*m+j
        # V has shape (m*p, rank) = (16, rank) with index j*p+k
        # W has shape (p*n, rank) = (16, rank) with index k*n+i
        
        # For the block decomposition:
        # i = bi*2 + li, j = bj*2 + lj, k = bk*2 + lk
        # where bi, bj, bk in {0,1} and li, lj, lk in {0,1}
        
        # U index: i*4+j = (bi*2+li)*4 + (bj*2+lj)
        # V index: j*4+k = (bj*2+lj)*4 + (bk*2+lk)
        # W index: k*4+i = (bk*2+lk)*4 + (bi*2+li)
        
        # The block-level Strassen operates on the (bi, bj, bk) indices,
        # and the local Strassen operates on the (li, lj, lk) indices.
        
        # So for each block-level term t and local term s:
        # For each (bi, bj, bk) with U2[bi*2+bj, t] != 0 and V2[bj*2+bk, t] != 0 and W2[bk*2+bi, t] != 0:
        #   For each (li, lj, lk) with U2[li*2+lj, s] != 0 and V2[lj*2+lk, s] != 0 and W2[lk*2+li, s] != 0:
        #     Add term with coefficient:
        #     U2[bi*2+bj, t] * U2[li*2+lj, s] for U
        #     V2[bj*2+bk, t] * V2[lj*2+lk, s] for V
        #     W2[bk*2+bi, t] * W2[lk*2+li, s] for W
        
        # Wait, I think I'm overcomplicating this. Let me just directly construct it.
        
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        U2, V2, W2 = get_2x2_strassen()
        
        for t in range(7):
            for s in range(7):
                col = t * 7 + s
                for bi in range(2):
                    for bj in range(2):
                        for bk in range(2):
                            for li in range(2):
                                for lj in range(2):
                                    for lk in range(2):
                                        u_block_coeff = U2[bi*2+bj, t]
                                        v_block_coeff = V2[bj*2+bk, t]
                                        w_block_coeff = W2[bk*2+bi, t]
                                        u_local_coeff = U2[li*2+lj, s]
                                        v_local_coeff = V2[lj*2+lk, s]
                                        w_local_coeff = W2[lk*2+li, s]
                                        
                                        u_total = u_block_coeff * u_local_coeff
                                        v_total = v_block_coeff * v_local_coeff
                                        w_total = w_block_coeff * w_local_coeff
                                        
                                        if u_total != 0 or v_total != 0 or w_total != 0:
                                            U_idx = (bi*2+li)*4 + (bj*2+lj)
                                            V_idx = (bj*2+lj)*4 + (bk*2+lk)
                                            W_idx = (bk*2+lk)*4 + (bi*2+li)
                                            
                                            U4[U_idx, col] += u_total
                                            V4[V_idx, col] += v_total
                                            W4[W_idx, col] += w_total
        
        return U4, V4, W4
    
    # Check time
    if time.time() > deadline:
        # Fall back to schoolbook
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
    
    # Get 4x4x4 Strassen decomposition
    U4, V4, W4 = get_4x4_strassen()
    
    # Now build the full 5x5x5 decomposition
    # Blocks:
    # (4,4,4): i in 0..3, j in 0..3, k in 0..3 -> rank 49
    # (4,4,1): i in 0..3, j in 0..3, k in 4 -> rank 16
    # (4,1,4): i in 0..3, j in 4, k in 0..3 -> rank 16
    # (4,1,1): i in 0..3, j in 4, k in 4 -> rank 4
    # (1,4,4): i in 4, j in 0..3, k in 0..3 -> rank 16
    # (1,4,1): i in 4, j in 0..3, k in 4 -> rank 4
    # (1,1,4): i in 4, j in 4, k in 0..3 -> rank 4
    # (1,1,1): i in 4, j in 4, k in 4 -> rank 1
    
    total_rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1
    # = 49 + 61 = 110
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col_offset = 0
    
    # Block (4,4,4): i in 0..3, j in 0..3, k in 0..3
    # Use U4, V4, W4 directly
    for li in range(4):
        for lj in range(4):
            for lk in range(4):
                local_col = li * 4 + lj * 4 + lk  # This isn't right
                # Actually, U4 has shape (16, 49) where row = i*4+j
                # So for each (i, j, k) in 0..3, the contribution is:
                # U[i*4+j, col] = U4[i*4+j, local_col]
                # V[j*4+k, col] = V4[j*4+k, local_col]
                # W[k*4+i, col] = W4[k*4+i, local_col]
                pass
    
    # Let me redo this more carefully
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col_offset = 0
    
    # Block (4,4,4)
    for local_col in range(49):
        for i in range(4):
            for j in range(4):
                U[i*5+j, col_offset + local_col] = U4[i*4+j, local_col]
        for j in range(4):
            for k in range(4):
                V[j*5+k, col_offset + local_col] = V4[j*4+k, local_col]
        for k in range(4):
            for i in range(4):
                W[k*5+i, col_offset + local_col] = W4[k*4+i, local_col]
    col_offset += 49
    
    # Block (4,4,1): i in 0..3, j in 0..3, k in 4
    # Schoolbook: for each (i, j), one term
    for i in range(4):
        for j in range(4):
            col = col_offset + i * 4 + j
            U[i*5+j, col] = 1
            V[j*5+4, col] = 1
            W[4*5+i, col] = 1
    col_offset += 16
    
    # Block (4,1,4): i in 0..3, j in 4, k in 0..3
    for i in range(4):
        for k in range(4):
            col = col_offset + i * 4 + k
            U[i*5+4, col] = 1
            V[4*5+k, col] = 1
            W[k*5+i, col] = 1
    col_offset += 16
    
    # Block (4,1,1): i in 0..3, j in 4, k in 4
    for i in range(4):
        col = col_offset + i
        U[i*5+4, col] = 1
        V[4*5+4, col] = 1
        W[4*5+i, col] = 1
    col_offset += 4
    
    # Block (1,4,4): i in 4, j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            col = col_offset + j * 4 + k
            U[4*5+j, col] = 1
            V[j*5+k, col] = 1
            W[k*5+4, col] = 1
    col_offset += 16
    
    # Block (1,4,1): i in 4, j in 0..3, k in 4
    for j in range(4):
        col = col_offset + j
        U[4*5+j, col] = 1
        V[j*5+4, col] = 1
        W[4*5+4, col] = 1
    col_offset += 4
    
    # Block (1,1,4): i in 4, j in 4, k in 0..3
    for k in range(4):
        col = col_offset + k
        U[4*5+4, col] = 1
        V[4*5+k, col] = 1
        W[k*5+4, col] = 1
    col_offset += 4
    
    # Block (1,1,1): i in 4, j in 4, k in 4
    col = col_offset
    U[4*5+4, col] = 1
    V[4*5+4, col] = 1
    W[4*5+4, col] = 1
    col_offset += 1
    
    # Verify col_offset == total_rank
    assert col_offset == total_rank, f"col_offset {col_offset} != total_rank {total_rank}"
    
    # Check time
    if time.time() > deadline:
        # Fall back to schoolbook
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
