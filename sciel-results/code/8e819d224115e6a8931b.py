"""Matrix multiplication 2x2 x 2x13 - tensor decomposition search."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen's 7-term decomposition for 2x2 * 2x2 blocks.
    # For 2x2 * 2x13, we have 6 full 2x2 blocks plus one 2x1 column.
    # We can group the 13 columns into 6 pairs (12 columns) + 1 remaining column.
    # Each pair of columns (2k, 2k+1) forms a 2x2 block with the 2x2 matrix.
    # Use Strassen rank-7 for each 2x2 block: 6 * 7 = 42 terms.
    # For the remaining single column (column 12), use schoolbook rank 2 (2 terms for 2x1).
    # Total rank = 42 + 2 = 44.
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # Strassen vectors for 2x2 * 2x2:
    # M1 = A11*(B11+B22) -> u=[1,0,0,0], v=[1,0,0,1]
    # M2 = (A11+A22)*B22 -> u=[1,0,0,1], v=[0,0,0,1]
    # M3 = (A12-A22)*(B21+B22) -> u=[0,1,0,-1], v=[0,0,1,1]
    # M4 = A21*(B11+B12) -> u=[0,0,1,0], v=[1,1,0,0]
    # M5 = (A11-A12)*B21 -> u=[1,-1,0,0], v=[0,0,1,0]
    # M6 = (A21+A22)*(B12-B22) -> u=[0,0,1,1], v=[0,1,-1,0]
    # M7 = (A11-A22)*(B11+B12) -> u=[1,0,0,-1], v=[1,1,0,0]
    
    # Index convention: A is 2x2, indices (i,j) -> i*m+j
    # B is 2x2, indices (j,k) -> j*p+k
    # For a 2x2 block at columns (2k, 2k+1), B block indices are (j, 2k) and (j, 2k+1)
    
    # Strassen U vectors (for A, 4 entries): [A11, A12, A21, A22]
    # Strassen V vectors (for B block, 4 entries): [B11, B12, B21, B22] for the 2x2 block
    # Strassen W vectors (for C, 4 entries): [C11, C12, C21, C22]
    
    # The 7 Strassen terms with their (u, v, w) vectors:
    # Each term contributes u_i * v_j * w_k to C[i,k]
    
    strassen_terms = [
        # (u_vec, v_vec, w_vec)
        # M1 = A11*(B11+B22)
        ([1,0,0,0], [1,0,0,1], [1,0,0,1]),
        # M2 = (A11+A22)*B22
        ([1,0,0,1], [0,0,0,1], [0,0,0,1]),
        # M3 = (A12-A22)*(B21+B22)
        ([0,1,0,-1], [0,0,1,1], [0,1,0,0]),
        # M4 = A21*(B11+B12)
        ([0,0,1,0], [1,1,0,0], [0,0,1,0]),
        # M5 = (A11-A12)*B21
        ([1,-1,0,0], [0,0,1,0], [1,0,0,0]),
        # M6 = (A21+A22)*(B12-B22)
        ([0,0,1,1], [0,1,-1,0], [0,1,0,0]),
        # M7 = (A11-A22)*(B11+B12)
        ([1,0,0,-1], [1,1,0,0], [1,0,0,0]),
    ]
    
    # Verify Strassen: C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's verify M1: u=[1,0,0,0], v=[1,0,0,1], w=[1,0,0,1]
    # M1 = A11 * (B11+B22)
    # C11 gets M1: A11*B11 + A11*B22... wait, w=[1,0,0,1] means C11 += M1 and C22 += M1
    # That's not right. Let me re-derive.
    
    # Actually, the standard Strassen is:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me use this standard formulation.
    strassen_terms = [
        # M1 = (A11+A22)*(B11+B22)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (A21+A22)*B11
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # M3 = A11*(B12-B22)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        # M4 = A22*(B21-B11)
        ([0,0,0,1], [0,0,1,-1], [0,0,1,0]),
        # M5 = (A11+A12)*B22
        ([1,1,0,0], [0,0,0,1], [1,0,0,0]),
        # M6 = (A21-A11)*(B11+B12)
        ([-1,0,1,0], [1,1,0,0], [0,1,0,0]),
        # M7 = (A12-A22)*(B21+B22)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Now build the full decomposition.
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (26, rank)
    # W has shape (p*n, rank) = (26, rank)
    
    # For each 2x2 block at columns (2k, 2k+1), k=0..5:
    # The B block has entries B[j, 2k] and B[j, 2k+1] for j=0,1
    # In the V vector, index j*p + (2k) and j*p + (2k+1)
    
    # For the last column (column 12), use schoolbook:
    # C[i, 12] = sum_j A[i,j] * B[j, 12]
    # This needs 2 terms (for j=0 and j=1)
    
    rank = 6 * 7 + 2  # 42 + 2 = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 6 Strassen blocks
    for k in range(6):
        for (u_vec, v_vec, w_vec) in strassen_terms:
            # u_vec has 4 entries for A[i,j], i=0,1; j=0,1
            # A index: i*m + j = i*2 + j
            for ij in range(4):
                i = ij // 2
                j = ij % 2
                U[i*m + j][col] = u_vec[ij]
            
            # v_vec has 4 entries for B[j, 2k] and B[j, 2k+1]
            # B index: j*p + col_offset
            # v_vec[0] = B[0, 2k], v_vec[1] = B[0, 2k+1], v_vec[2] = B[1, 2k], v_vec[3] = B[1, 2k+1]
            for jj in range(4):
                j = jj // 2
                col_offset = jj % 2
                b_idx = j * p + (2*k + col_offset)
                V[b_idx][col] = v_vec[jj]
            
            # w_vec has 4 entries for C[i, 2k] and C[i, 2k+1]
            # C index: (2k)*n + i and (2k+1)*n + i
            # w_vec[0] = C[0, 2k], w_vec[1] = C[0, 2k+1], w_vec[2] = C[1, 2k], w_vec[3] = C[1, 2k+1]
            for ii in range(4):
                i = ii // 2
                col_offset = ii % 2
                c_idx = (2*k + col_offset) * n + i
                W[c_idx][col] = w_vec[ii]
            
            col += 1
    
    # Schoolbook for last column (column 12)
    for j in range(m):
        for i in range(n):
            U[i*m + j][col] = 1
            V[j*p + 12][col] = 1
            W[12*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
