"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base Strassen decomposition for 2x2x2 matrices
    # Tensors for M_{2,2,2} have rank 7
    # Indices are (i, j, k) for 0<=i,j,k<2
    # We define U, V, W such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = (A B)_{i,k}
    
    # The 7 terms of Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will build the decomposition for M_{2,2,2}
    # U has shape (2*2, 7) -> (4, 7) where index is i*2+j
    # V has shape (2*2, 7) -> (4, 7) where index is j*2+k
    # W has shape (2*2, 7) -> (4, 7) where index is k*2+i
    
    U_2 = [[0]*7 for _ in range(4)]
    V_2 = [[0]*7 for _ in range(4)]
    W_2 = [[0]*7 for _ in range(4)]
    
    # c=0: M1
    # (A00 + A11)
    U_2[0][0] = 1 # i=0, j=0
    U_2[3][0] = 1 # i=1, j=1
    # (B00 + B11)
    V_2[0][0] = 1 # j=0, k=0
    V_2[3][0] = 1 # j=1, k=1
    # (C00 + C11)
    W_2[0][0] = 1 # k=0, i=0
    W_2[3][0] = 1 # k=1, i=1
    
    # c=1: M2
    # (A10 + A11)
    U_2[2][1] = 1 # i=1, j=0
    U_2[3][1] = 1 # i=1, j=1
    # B00
    V_2[0][1] = 1 # j=0, k=0
    # C10
    W_2[2][1] = 1 # k=0, i=1
    
    # c=2: M3
    # A00
    U_2[0][2] = 1 # i=0, j=0
    # (B01 - B11)
    V_2[1][2] = 1 # j=0, k=1
    V_2[3][2] = -1 # j=1, k=1
    # C01
    W_2[1][2] = 1 # k=1, i=0
    
    # c=3: M4
    # A11
    U_2[3][3] = 1 # i=1, j=1
    # (B10 - B00)
    V_2[2][3] = 1 # j=1, k=0
    V_2[0][3] = -1 # j=0, k=0
    # (C00 + C10)
    W_2[0][3] = 1 # k=0, i=0
    W_2[2][3] = 1 # k=0, i=1
    
    # c=4: M5
    # (A00 + A01)
    U_2[0][4] = 1 # i=0, j=0
    U_2[1][4] = 1 # i=0, j=1
    # B11
    V_2[3][4] = 1 # j=1, k=1
    # (C00 + C01)
    W_2[0][4] = 1 # k=0, i=0
    W_2[1][4] = 1 # k=1, i=0
    
    # c=5: M6
    # (A10 - A00)
    U_2[2][5] = 1 # i=1, j=0
    U_2[0][5] = -1 # i=0, j=0
    # (B00 + B01)
    V_2[0][5] = 1 # j=0, k=0
    V_2[1][5] = 1 # j=0, k=1
    # C11
    W_2[3][5] = 1 # k=1, i=1
    
    # c=6: M7
    # (A01 - A11)
    U_2[1][6] = 1 # i=0, j=1
    U_2[3][6] = -1 # i=1, j=1
    # (B10 + B11)
    V_2[2][6] = 1 # j=1, k=0
    V_2[3][6] = 1 # j=1, k=1
    # (C00 + C11)
    W_2[0][6] = 1 # k=0, i=0
    W_2[3][6] = 1 # k=1, i=1
    
    # Now we recursively apply this to build larger decompositions.
    # We need M_{4,8,8} essentially, but we can just build M_{8,8,8} and extract.
    # Actually, building M_{4,4,4} then M_{8,8,8} is easier.
    
    def expand(U_in, V_in, W_in, size_in):
        # size_in is 2, 4, 8...
        # We split indices into high and low parts? No, standard Kronecker.
        # If we have a decomposition for M_{N,N,N} with rank R.
        # We can construct M_{2N,2N,2N} with rank 7*R.
        # The new U has shape (4N, 7R)
        # The new V has shape (4N, 7R)
        # The new W has shape (4N, 7R)
        
        R = len(U_in[0])
        U_new = [[0]*(7*R) for _ in range(4*size_in)]
        V_new = [[0]*(7*R) for _ in range(4*size_in)]
        W_new = [[0]*(7*R) for _ in range(4*size_in)]
        
        for c in range(7):
            # Map the c-th Strassen term to the new indices
            # The 7 terms correspond to the coefficients in the expansion.
            # Let's define the mapping from (i', j') to (i, j) for the blocks.
            # i' = 2*i + d_i ? No.
            # The matrix A is 2N x 2N.
            # A = [[A00, A01], [A10, A11]] where blocks are N x N.
            # The decomposition of A*B = C uses 7 block multiplications.
            # The indices for the new U rows are (i, j) where 0<=i,j<2N.
            # We can map (i, j) to (block_i, block_j, local_i, local_j).
            # block_i = i // N, local_i = i % N.
            # Similarly for j.
            
            # For a given Strassen term c (0..6), we need to know:
            # 1. Which combinations of A blocks are used.
            # 2. Which combinations of B blocks are used.
            # 3. Which combinations of C blocks are produced.
            
            # Let's predefine the block coefficients for U, V, W for each c.
            # U_c acts on A. V_c acts on B. W_c acts on C.
            # The entries in U_new[row_i, col_c] are non-zero only if (i,j) is in the support of U_c.
            
            # Let's define the support and coefficients.
            # c=0: M1 = (A00+A11)(B00+B11)
            # U support: (0,0), (1,1). Coeff: 1, 1.
            # V support: (0,0), (1,1). Coeff: 1, 1.
            # W support: (0,0), (1,1). Coeff: 1, 1.
            
            # c=1: M2 = (A10+A11)B00
            # U support: (1,0), (1,1). Coeff: 1, 1.
            # V support: (0,0). Coeff: 1.
            # W support: (1,0). Coeff: 1.
            
            # c=2: M3 = A00(B01-B11)
            # U support: (0,0). Coeff: 1.
            # V support: (0,1), (1,1). Coeff: 1, -1.
            # W support: (0,1). Coeff: 1.
            
            # c=3: M4 = A11(B10-B00)
            # U support: (1,1). Coeff: 1.
            # V support: (1,0), (0,0). Coeff: 1, -1.
            # W support: (0,0), (1,0). Coeff: 1, 1.
            
            # c=4: M5 = (A00+A01)B11
            # U support: (0,0), (0,1). Coeff: 1, 1.
            # V support: (1,1). Coeff: 1.
            # W support: (0,0), (0,1). Coeff: 1, 1.
            
            # c=5: M6 = (A10-A00)(B00+B01)
            # U support: (1,0), (0,0). Coeff: 1, -1.
            # V support: (0,0), (0,1). Coeff: 1, 1.
            # W support: (1,1). Coeff: 1.
            
            # c=6: M7 = (A01-A11)(B10+B11)
            # U support: (0,1), (1,1). Coeff: 1, -1.
            # V support: (1,0), (1,1). Coeff: 1, 1.
            # W support: (0,0), (1,1). Coeff: 1, 1.
            
            # We define these as lists of (block_row, block_col, coeff)
            U_blocks = [
                [(0,0,1), (1,1,1)],
                [(1,0,1), (1,1,1)],
                [(0,0,1)],
                [(1,1,1)],
                [(0,0,1), (0,1,1)],
                [(1,0,1), (0,0,-1)],
                [(0,1,1), (1,1,-1)]
            ]
            V_blocks = [
                [(0,0,1), (1,1,1)],
                [(0,0,1)],
                [(0,1,1), (1,1,-1)],
                [(1,0,1), (0,0,-1)],
                [(1,1,1)],
                [(0,0,1), (0,1,1)],
                [(1,0,1), (1,1,1)]
            ]
            W_blocks = [
                [(0,0,1), (1,1,1)],
                [(1,0,1)],
                [(0,1,1)],
                [(0,0,1), (1,0,1)],
                [(0,0,1), (0,1,1)],
                [(1,1,1)],
                [(0,0,1), (1,1,1)]
            ]
            
            for local_i in range(size_in):
                for local_j in range(size_in):
                    for bi, bj, coeff_u in U_blocks[c]:
                        new_i = bi * size_in + local_i
                        new_j = bj * size_in + local_j
                        row_u = new_i * (2*size_in) + new_j
                        # The column in the new U is c*R + k, where k is the local column
                        # We need to map the local (local_j, local_k) to the column.
                        # Actually, the inner product is over the local indices.
                        # U_new[row_u, col] = coeff_u * U_in[local_i*size_in+local_j, k] ?
                        # No. U_in has rows (i, j) and columns c'.
                        # The new column index is c * R + c'.
                        # So for a fixed c, we iterate c' from 0 to R-1.
                        # U_new[row_u, c*R + c'] += coeff_u * U_in[local_i*size_in+local_j, c']
                        
                        for c_prime in range(R):
                            U_new[row_u][c*R + c_prime] += coeff_u * U_in[local_i*size_in + local_j][c_prime]
                        
            for local_j in range(size_in):
                for local_k in range(size_in):
                    for bj, bk, coeff_v in V_blocks[c]:
                        new_j = bj * size_in + local_j
                        new_k = bk * size_in + local_k
                        row_v = new_j * (2*size_in) + new_k
                        for c_prime in range(R):
                            V_new[row_v][c*R + c_prime] += coeff_v * V_in[local_j*size_in + local_k][c_prime]
                        
            for local_k in range(size_in):
                for local_i in range(size_in):
                    for bk, bi, coeff_w in W_blocks[c]:
                        new_k = bk * size_in + local_k
                        new_i = bi * size_in + local_i
                        row_w = new_k * (2*size_in) + new_i
                        for c_prime in range(R):
                            W_new[row_w][c*R + c_prime] += coeff_w * W_in[local_k*size_in + local_i][c_prime]
                            
        return U_new, V_new, W_new

    # Start with 2x2x2
    U_cur = U_2
    V_cur = V_2
    W_cur = W_2
    size_cur = 2
    
    # Expand to 4x4x4
    U_cur, V_cur, W_cur = expand(U_cur, V_cur, W_cur, size_cur)
    size_cur = 4
    
    # Expand to 8x8x8
    U_cur, V_cur, W_cur = expand(U_cur, V_cur, W_cur, size_cur)
    size_cur = 8
    
    # Now we have a decomposition for M_{8,8,8} with rank 49.
    # We need to extract the sub-tensor for M_{2,5,6}.
    # The input matrices are A (2x5), B (5x6).
    # We pad them to 8x8.
    # A_padded is 8x8. A[i,j] = A_orig[i,j] if i<2, j<5 else 0.
    # B_padded is 8x8. B[i,j] = B_orig[i,j] if i<5, j<6 else 0.
    # C_padded is 8x8. C[i,j] = C_orig[i,j] if i<2, j<6 else 0.
    
    # The tensor decomposition is for the full 8x8x8 multiplication.
    # We only care about the terms that contribute to C[i,j] for i<2, j<6.
    # The contraction is sum_{k=0}^7 sum_{r=0}^{48} U[i, k, r] V[k, j, r] W[j, i, r] = C[i, j].
    # Wait, the indices in U, V, W are flattened.
    # U has rows (i, k) where i, k are 0..7. Shape (64, 49).
    # V has rows (k, j) where k, j are 0..7. Shape (64, 49).
    # W has rows (j, i) where j, i are 0..7. Shape (64, 49).
    
    # We want to restrict i to [0,2), k to [0,5), j to [0,6).
    # Note: In the problem, A is 2x5, B is 5x6.
    # A_{i,k}, B_{k,j}.
    # So k is the shared dimension, which is 5.
    # So we restrict k < 5.
    # i < 2.
    # j < 6.
    
    # We construct new U, V, W matrices.
    # New U will have rows corresponding to (i, k) where i<2, k<5.
    # New V will have rows corresponding to (k, j) where k<5, j<6.
    # New W will have rows corresponding to (j, i) where j<6, i<2.
    
    # The columns (rank) remain the same, 49.
    # We just select the relevant rows from U_cur, V_cur, W_cur.
    
    # U_cur row index is i*8 + k.
    # We need rows where i<2 and k<5.
    U_rows = []
    for i in range(2):
        for k in range(5):
            U_rows.append(i*8 + k)
            
    # V_cur row index is k*8 + j.
    # We need rows where k<5 and j<6.
    V_rows = []
    for k in range(5):
        for j in range(6):
            V_rows.append(k*8 + j)
            
    # W_cur row index is j*8 + i.
    # We need rows where j<6 and i<2.
    W_rows = []
    for j in range(6):
        for i in range(2):
            W_rows.append(j*8 + i)
            
    U_new = [U_cur[r] for r in U_rows]
    V_new = [V_cur[r] for r in V_rows]
    W_new = [W_cur[r] for r in W_rows]
    
    # The rank is still 49.
    # We can potentially remove zero columns to reduce rank, but for now we return the full 49.
    # Actually, we should check if any columns are all zeros and remove them to minimize rank.
    
    rank = len(U_new[0])
    
    # Find non-zero columns
    non_zero_cols = []
    for c in range(rank):
        col_u = [U_new[r][c] for r in range(len(U_new))]
        col_v = [V_new[r][c] for r in range(len(V_new))]
        col_w = [W_new[r][c] for r in range(len(W_new))]
        if any(x != 0 for x in col_u) or any(x != 0 for x in col_v) or any(x != 0 for x in col_w):
            non_zero_cols.append(c)
            
    # Filter columns
    U_final = [[U_new[r][c] for c in non_zero_cols] for r in range(len(U_new))]
    V_final = [[V_new[r][c] for c in non_zero_cols] for r in range(len(V_new))]
    W_final = [[W_new[r][c] for c in non_zero_cols] for r in range(len(W_new))]
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
