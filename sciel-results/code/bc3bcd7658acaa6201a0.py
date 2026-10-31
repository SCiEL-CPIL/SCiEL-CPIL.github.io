"""Program for 4x4 matrix multiplication tensor decomposition.
Uses the rank-49 Strassen-based decomposition (two levels of rank-7).
"""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    The decomposition uses two levels of Strassen's algorithm:
    - Level 1: 4x4 = blocks of 2x2, using 7 multiplications of 2x2 blocks
    - Level 2: each 2x2 multiplication uses rank-7 Strassen (7 multiplications of scalars)
    Total rank: 7 * 7 = 49
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Indices: A[i,j] with i,j in {0,1}; B[i,j] with i,j in {0,1}
    # U2: 4 x 7, V2: 4 x 7, W2: 4 x 7
    # U2[i*m+j, r] = coefficient of A[i,j] in the r-th multiplication
    # V2[j*p+k, r] = coefficient of B[j,k] in the r-th multiplication
    # W2[k*n+i, r] = coefficient for C[i,k] from the r-th multiplication
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2: indices are A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # W convention: W[k*n+i, r] for 2x2 means W[k*2+i, r]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U2[0][0] = 1  # A[0,0]
    U2[3][0] = 1  # A[1,1]
    V2[0][0] = 1  # B[0,0]
    V2[3][0] = 1  # B[1,1]
    W2[0][0] = 1  # C[0,0]: k=0,i=0
    W2[3][0] = 1  # C[1,1]: k=1,i=1
    
    # M2: (A10+A11)B00
    U2[2][1] = 1  # A[1,0]
    U2[3][1] = 1  # A[1,1]
    V2[0][1] = 1  # B[0,0]
    W2[2][1] = 1  # C[1,0]: k=0,i=1
    
    # M3: A00(B01-B11)
    U2[0][2] = 1  # A[0,0]
    V2[1][2] = 1  # B[0,1]
    V2[3][2] = -1  # B[1,1]
    W2[1][2] = 1  # C[0,1]: k=1,i=0
    
    # M4: A11(B10-B00)
    U2[3][3] = 1  # A[1,1]
    V2[2][3] = 1  # B[1,0]
    V2[0][3] = -1  # B[0,0]
    W2[0][3] = 1  # C[0,0]: k=0,i=0
    W2[2][3] = 1  # C[1,0]: k=0,i=1
    
    # M5: (A00+A01)B11
    U2[0][4] = 1  # A[0,0]
    U2[1][4] = 1  # A[0,1]
    V2[3][4] = 1  # B[1,1]
    W2[0][4] = -1  # C[0,0]: k=0,i=0
    W2[1][4] = 1  # C[0,1]: k=1,i=0
    
    # M6: (A10-A00)(B00+B01)
    U2[2][5] = 1  # A[1,0]
    U2[0][5] = -1  # A[0,0]
    V2[0][5] = 1  # B[0,0]
    V2[1][5] = 1  # B[0,1]
    W2[3][5] = 1  # C[1,1]: k=1,i=1
    
    # M7: (A01-A11)(B10+B11)
    U2[1][6] = 1  # A[0,1]
    U2[3][6] = -1  # A[1,1]
    V2[2][6] = 1  # B[1,0]
    V2[3][6] = 1  # B[1,1]
    W2[0][6] = 1  # C[0,0]: k=0,i=0
    
    # Now build the 4x4 decomposition using Strassen at the block level.
    # 4x4 matrices are divided into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Global indices for 4x4: A[i,j] with i,j in {0,1,2,3}
    # Block A00: rows 0-1, cols 0-1 -> global indices 0,1,4,5
    # Block A01: rows 0-1, cols 2-3 -> global indices 2,3,6,7
    # Block A10: rows 2-3, cols 0-1 -> global indices 8,9,12,13
    # Block A11: rows 2-3, cols 2-3 -> global indices 10,11,14,15
    #
    # Similarly for B and C.
    #
    # Strassen at block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mr is a product of two 2x2 matrices, decomposed via rank-7 U2,V2,W2.
    # For each block-level multiplication Mr, we need to:
    # 1. Form the linear combinations of A-blocks (using U coefficients)
    # 2. Form the linear combinations of B-blocks (using V coefficients)
    # 3. Multiply them using the 2x2 rank-7 decomposition
    # 4. Add the result to the appropriate C-blocks (using W coefficients)
    
    # Block index to global row/col mapping for 4x4:
    # Block (bi, bj) where bi,bj in {0,1} corresponds to rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    
    def block_to_global(r_block, c_block):
        """Map block-local (r,c) in 2x2 block to global index in 4x4 (16 entries)."""
        return (r_block * 2 + c_block) * 4 + c_block  # Wait, this isn't right
        # Actually for 4x4: global index = row*4 + col
        # Block (bi,bj) local (r,c) -> global row = 2*bi + r, global col = 2*bj + c
        # So global index = (2*bi+r)*4 + (2*bj+c)
    
    # Let me redo this more carefully.
    # U is 16 x 49, V is 16 x 49, W is 16 x 49
    # U[i,j, r] where i is row of A (0-3), j is col of A (0-3), so global index = i*4+j
    # V[j,k, r] where j is row of B (0-3), k is col of B (0-3), global index = j*4+k
    # W[k,i, r] where k is col of C (0-3), i is row of C (0-3), global index = k*4+i (per task convention W[k*n+i])
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # For each of the 7 block-level Strassen products, we have 7 inner products.
    # Total: 7*7 = 49 terms.
    
    # Define the block-level Strassen coefficients
    # For product r (0-6):
    # A-blocks involved with coefficients
    # B-blocks involved with coefficients
    
    # A_block[r][block_index] = coefficient
    # B_block[r][block_index] = coefficient
    # C_block[r][block_index] = coefficient (for output)
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (same for B and C)
    
    A_coeff = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [1, 0, 1, 0],  # M6: A10 - A00 -> [-1, 0, 1, 0]
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    # Fix M6: A10 - A00
    A_coeff[5] = [-1, 0, 1, 0]
    
    B_coeff = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00 -> [0,0,-1,1]
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    # Fix M4: B10 - B00
    B_coeff[3] = [0, 0, -1, 1]
    
    C_coeff = [
        [1, 0, 0, 0, 0, 0, 0, 1],  # M1: C00 + C11 -> blocks 0 and 3
        [0, 0, 1, 0, 0, 0, 0, 0],  # M2: C10 -> block 2
        [0, 0, 0, 0, 1, 0, 0, 0],  # M3: C01 -> block 1
        [1, 0, 1, 0, 0, 0, 0, 0],  # M4: C00 + C10 -> blocks 0 and 2
        [-1, 1, 0, 0, 0, 0, 0, 0], # M5: -C00 + C01 -> blocks 0 and 1
        [0, 0, 0, 0, 0, 0, 0, 1],  # M6: C11 -> block 3
        [1, 0, 0, 0, 0, 0, 0, 0],  # M7: C00 -> block 0
    ]
    # Let me redefine C_coeff more carefully as a list of (block_index, coefficient)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block-level product r, it contributes to certain C-blocks with certain signs.
    # c_contrib[r][c_block] = coefficient
    
    c_contrib = [
        {0: 1, 3: 1},    # M1 -> C00 (+1), C11 (+1)
        {2: 1},           # M2 -> C10 (+1)
        {1: 1},           # M3 -> C01 (+1)
        {0: 1, 2: 1},    # M4 -> C00 (+1), C10 (+1)
        {0: -1, 1: 1},   # M5 -> C00 (-1), C01 (+1)
        {3: 1},           # M6 -> C11 (+1)
        {0: 1},           # M7 -> C00 (+1)
    ]
    
    # Now for each block-level product r (0-6) and inner product s (0-6):
    # The term index is t = r*7 + s
    # U contribution: for each A-block b_a with coefficient a_c, and for each local (i,j) in that block,
    #   U[global_i*4+global_j, t] += a_c * U2[i*2+j, s]
    # V contribution: for each B-block b_b with coefficient b_c, and for each local (j,k) in that block,
    #   V[global_j*4+global_k, t] += b_c * V2[j*2+k, s]
    # W contribution: for each C-block b_c with coefficient c_c, and for each local (i,k) in that block,
    #   W[global_k*4+global_i, t] += c_c * W2[k*2+i, s]
    
    for r in range(7):
        for s in range(7):
            t = r * 7 + s
            
            # U contributions
            for b_a in range(4):
                a_c = A_coeff[r][b_a]
                if a_c == 0:
                    continue
                # Block b_a: bi = b_a // 2, bj = b_a % 2
                bi, bj = b_a // 2, b_a % 2
                for li in range(2):
                    for lj in range(2):
                        # Local index in 2x2 block: li*2 + lj
                        local_idx = li * 2 + lj
                        u2_val = U2[local_idx][s]
                        if u2_val == 0:
                            continue
                        # Global indices: row = 2*bi + li, col = 2*bj + lj
                        gi = 2 * bi + li
                        gj = 2 * bj + lj
                        global_idx = gi * 4 + gj
                        U[global_idx][t] += a_c * u2_val
            
            # V contributions
            for b_b in range(4):
                b_c = B_coeff[r][b_b]
                if b_c == 0:
                    continue
                bi, bj = b_b // 2, b_b % 2
                for lj in range(2):
                    for lk in range(2):
                        local_idx = lj * 2 + lk
                        v2_val = V2[local_idx][s]
                        if v2_val == 0:
                            continue
                        gi = 2 * bi + lj
                        gj = 2 * bj + lk
                        global_idx = gi * 4 + gj
                        V[global_idx][t] += b_c * v2_val
            
            # W contributions
            for c_block, c_c in c_contrib[r].items():
                if c_c == 0:
                    continue
                bi, bj = c_block // 2, c_block % 2
                for li in range(2):
                    for lk in range(2):
                        local_idx = lk * 2 + li  # W2[k*n+i] convention: k is row of local block, i is col
                        w2_val = W2[local_idx][s]
                        if w2_val == 0:
                            continue
                        gi = 2 * bi + li  # row of C
                        gj = 2 * bj + lk  # col of C
                        # W convention: W[k*n+i] where k is col index, i is row index
                        # So W[global_k*4 + global_i]
                        global_idx = gj * 4 + gi
                        W[global_idx][t] += c_c * w2_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
