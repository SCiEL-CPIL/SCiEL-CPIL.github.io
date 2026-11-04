import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition block by block.
    # A is n x m (3x5)
    # B is m x p (5x7)
    # C is n x p (3x7)
    
    # We use a list of blocks. Each block is defined by:
    # i_start, i_end, j_start, j_end, k_start, k_end, rank, type
    # type: 'strassen' for 2x2x2, 'schoolbook' otherwise
    
    # To be safe and "new", I will implement a general schoolbook first,
    # then replace specific 2x2x2 blocks with Strassen.
    
    # Initialize with Schoolbook
    rank_total = n * m * p
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Schoolbook decomposition
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Term (i,j,k) maps to rank index r = i*m*p + j*p + k
    # U[i*m+j][r] = 1
    # V[j*p+k][r] = 1
    # W[k*n+i][r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    # Now, we try to replace 2x2x2 blocks with Strassen (rank 7) to save rank.
    # We need to identify disjoint 2x2x2 blocks in the tensor.
    # A rows: 0-1, 2
    # A cols: 0-1, 2-3, 4
    # B cols: 0-1, 2-3, 4-5, 6
    
    # Possible 2x2x2 blocks:
    # A[0:2, 0:2] * B[0:2, 0:2]
    # A[0:2, 0:2] * B[0:2, 2:4]
    # A[0:2, 0:2] * B[0:2, 4:6]
    # A[0:2, 2:4] * B[2:4, 0:2]
    # A[0:2, 2:4] * B[2:4, 2:4]
    # A[0:2, 2:4] * B[2:4, 4:6]
    
    # We will replace these 6 blocks with Strassen.
    # Each replacement saves 1 rank (8 -> 7).
    # Total save: 6.
    # New rank: 105 - 6 = 99.
    
    # To implement this "new" way, we will zero out the schoolbook terms for these blocks
    # and add the Strassen terms.
    
    def apply_strassen(i0, j0, k0, rank_offset):
        # This function assumes the rank_offset points to 7 unused slots
        # and zeroes out the 8 schoolbook slots that would have been used
        # if we were doing schoolbook for this 2x2x2 block.
        
        # Schoolbook indices for this block:
        # i in [i0, i0+1], j in [j0, j0+1], k in [k0, k0+1]
        # r = i*m*p + j*p + k
        
        # We need to clear these 8 entries in U, V, W
        # And then fill rank_offset to rank_offset+6 with Strassen values.
        
        # Strassen coefficients for 2x2x2
        # U_loc (4x7), V_loc (4x7), W_loc (4x7)
        # Indices in U_loc: i_l*2 + j_l
        # Indices in V_loc: j_l*2 + k_l
        # Indices in W_loc: k_l*2 + i_l
        
        u_loc = [
            [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
            [0, 0, 0, 0, 1, 0, 1], # i=0, j=1
            [0, 1, 0, 0, 0, 1, 0], # i=1, j=0
            [1, 1, 0, 1, 0, 0, -1] # i=1, j=1
        ]
        v_loc = [
            [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
            [0, 0, 1, 0, 0, 1, 0], # j=0, k=1
            [0, 0, 0, 1, 0, 0, 1], # j=1, k=0
            [1, 0, -1, 0, 1, 0, 1] # j=1, k=1
        ]
        w_loc = [
            [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
            [0, 1, 0, 1, 0, 0, 0], # k=0, i=1
            [0, 0, 1, 0, 1, 0, 0], # k=1, i=0
            [1, -1, 1, 0, 0, 1, 0] # k=1, i=1
        ]
        
        # Zero out schoolbook terms
        for i_l in range(2):
            for j_l in range(2):
                for k_l in range(2):
                    r_school = (i0 + i_l) * m * p + (j0 + j_l) * p + (k0 + k_l)
                    u_idx = (i0 + i_l) * m + (j0 + j_l)
                    v_idx = (j0 + j_l) * p + (k0 + k_l)
                    w_idx = (k0 + k_l) * n + (i0 + i_l)
                    
                    U[u_idx][r_school] = 0
                    V[v_idx][r_school] = 0
                    W[w_idx][r_school] = 0
        
        # Fill Strassen terms
        for i_l in range(2):
            for j_l in range(2):
                u_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(2):
                v_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        
        for k_l in range(2):
            for i_l in range(2):
                w_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]

    # We need to compact the rank.
    # The schoolbook rank is 105.
    # We are replacing 8 slots with 7 slots.
    # This requires shifting indices.
    
    # Instead of in-place modification with shifting, it's safer to build from scratch
    # using the block logic, similar to the current program but with a different structure.
    
    # Let's revert to building from scratch to avoid index shifting bugs.
    
    U = [[0] * 99 for _ in range(n * m)]
    V = [[0] * 99 for _ in range(m * p)]
    W = [[0] * 99 for _ in range(p * n)]
    
    rank_offset = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    apply_strassen(0, 0, 0, rank_offset)
    rank_offset += 7
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    apply_strassen(0, 0, 2, rank_offset)
    rank_offset += 7
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    apply_strassen(0, 0, 4, rank_offset)
    rank_offset += 7
    
    # Block 4: A[0:2, 0:2] * B[0:2, 6:7] (Schoolbook 2x2x1)
    for i_l in range(2):
        for j_l in range(2):
            for k_l in range(1):
                r = rank_offset + i_l*2 + j_l*1 + k_l # 4 terms
                u_idx = (0 + i_l) * m + (0 + j_l)
                v_idx = (0 + j_l) * p + (6 + k_l)
                w_idx = (6 + k_l) * n + (0 + i_l)
                U[u_idx][r] += 1
                V[v_idx][r] += 1
                W[w_idx][r] += 1
    rank_offset += 4
    
    # Block 5: A[2:3, 0:2] * B[0:2, 0:7] (Schoolbook 1x2x7)
    for i_l in range(1):
        for j_l in range(2):
            for k_l in range(7):
                r = rank_offset + i_l*14 + j_l*7 + k_l
                u_idx = (2 + i_l) * m + (0 + j_l)
                v_idx = (0 + j_l) * p + (0 + k_l)
                w_idx = (0 + k_l) * n + (2 + i_l)
                U[u_idx][r] += 1
                V[v_idx][r] += 1
                W[w_idx][r] += 1
    rank_offset += 14
    
    # Block 6: A[0:2, 2:4] * B[2:4, 0:2]
    apply_strassen(0, 2, 0, rank_offset)
    rank_offset += 7
    
    # Block 7: A[0:2, 2:4] * B[2:4, 2:4]
    apply_strassen(0, 2, 2, rank_offset)
    rank_offset += 7
    
    # Block 8: A[0:2, 2:4] * B[2:4, 4:6]
    apply_strassen(0, 2, 4, rank_offset)
    rank_offset += 7
    
    # Block 9: A[0:2, 2:4] * B[2:4, 6:7] (Schoolbook 2x2x1)
    for i_l in range(2):
        for j_l in range(2):
            for k_l in range(1):
                r = rank_offset + i_l*2 + j_l*1 + k_l
                u_idx = (0 + i_l) * m + (2 + j_l)
                v_idx = (2 + j_l) * p + (6 + k_l)
                w_idx = (6 + k_l) * n + (0 + i_l)
                U[u_idx][r] += 1
                V[v_idx][r] += 1
                W[w_idx][r] += 1
    rank_offset += 4
    
    # Block 10: A[2:3, 2:4] * B[2:4, 0:7] (Schoolbook 1x2x7)
    for i_l in range(1):
        for j_l in range(2):
            for k_l in range(7):
                r = rank_offset + i_l*14 + j_l*7 + k_l
                u_idx = (2 + i_l) * m + (2 + j_l)
                v_idx = (2 + j_l) * p + (0 + k_l)
                w_idx = (0 + k_l) * n + (2 + i_l)
                U[u_idx][r] += 1
                V[v_idx][r] += 1
                W[w_idx][r] += 1
    rank_offset += 14
    
    # Block 11: A[0:3, 4:5] * B[4:5, 0:7] (Schoolbook 3x1x7)
    for i_l in range(3):
        for j_l in range(1):
            for k_l in range(7):
                r = rank_offset + i_l*7 + j_l*7 + k_l
                u_idx = (0 + i_l) * m + (4 + j_l)
                v_idx = (4 + j_l) * p + (0 + k_l)
                w_idx = (0 + k_l) * n + (0 + i_l)
                U[u_idx][r] += 1
                V[v_idx][r] += 1
                W[w_idx][r] += 1
    rank_offset += 21
    
    return {"U": U, "V": V, "W": W}
