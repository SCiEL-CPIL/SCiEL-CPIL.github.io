import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    # Strassen multiplications:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1 # A00
    U[3][0] = 1 # A11
    V[0][0] = 1 # B00
    V[1][0] = 1 # B11
    W[0][0] = 1 # C00
    W[1][0] = -1 # C01? No.
    # Let's map carefully.
    # C00 gets M1. W[0*2+0][0] = 1.
    # C01 gets M5? No.
    
    # Let's define the contributions to C elements.
    # C00: M1, M4, -M5, M7
    # C01: M3, M5
    # C10: M2, M4
    # C11: M1, -M2, M3, M6
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    W[3][6] = 1  # C11
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Builds 4x4x4 decomposition using Strassen on 2x2 blocks.
    Rank 49.
    """
    n, m, p = 4, 4, 4
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # 4x4 matrix can be viewed as 2x2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # We apply Strassen to the block multiplication.
    # The 7 block multiplications are 2x2x2 multiplications.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank 7*7 = 49.
    
    # We need to map the scalar indices.
    # A indices: 0..15. Block (bi, bj) has indices 4*bi + 2*ri + cj?
    # Let's use a helper to get flat index.
    # A is 4x4. Index (i,j).
    # Block bi, bj. i = 2*bi + ri, j = 2*bj + cj.
    # Flat index = i*4 + j.
    
    # We generate the 7 Strassen terms for the block level.
    # For each block-level term r (0..6), we have a 2x2x2 decomposition.
    # The 2x2x2 decomposition provides 7 scalar multiplications.
    
    # Strassen block definitions (bi, bj, ck)
    # M1: A00+A11, B00+B11 -> C00, C11
    # M2: A10+A11, B00 -> C10, C11
    # M3: A00, B01-B11 -> C01, C11
    # M4: A11, B10-B00 -> C00, C10
    # M5: A00+A01, B11 -> C00, C01
    # M6: A10-A00, B00+B01 -> C11
    # M7: A01-A11, B10+B11 -> C00, C11
    
    # We will iterate over the 7 block terms and their 7 scalar terms.
    
    # Define the block-level U, V, W coefficients for the 7 terms.
    # U_block[bi][r], V_block[bj][r], W_block[ck][r]
    
    U_block = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, -1, -1] # A11
    ]
    
    V_block = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0], # B01
        [0, 0, 0, 1, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1] # B11
    ]
    
    W_block = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 1] # C11
    ]
    
    # Get 2x2x2 Strassen U, V, W
    U2, V2, W2 = _build_strassen_2x2x2()
    
    for r_block in range(7):
        for r_scalar in range(7):
            r = r_block * 7 + r_scalar
            
            # U2[r_scalar] is a vector of 4 elements (for 2x2 A block)
            # V2[r_scalar] is a vector of 4 elements (for 2x2 B block)
            # W2[r_scalar] is a vector of 4 elements (for 2x2 C block)
            
            # Map to 4x4 indices
            for bi in range(2):
                for ri in range(2):
                    # A index: bi, ri in block -> i = 2*bi + ri
                    # The scalar term r_scalar in U2 corresponds to a specific (ri, cj) in the 2x2 block?
                    # U2 is indexed by flat index of 2x2: i2*2 + j2.
                    # U2[r_scalar] has 1s at positions corresponding to the linear combination.
                    
                    # Let's iterate over the 4 positions in the 2x2 block
                    for pos in range(4):
                        if U2[pos][r_scalar] != 0:
                            i2 = pos // 2
                            j2 = pos % 2
                            i = 2*bi + i2
                            j = 2*bj + j2 # Wait, bj is not in loop.
                            
            # This nested loop is getting complicated. Let's simplify.
            # For each block term r_block, we have coefficients for A blocks, B blocks, C blocks.
            # U_block[bi][r_block] is the coefficient for A block (bi, bj)? No, U_block is defined per row of blocks.
            # Actually, U_block[bi][r_block] is the coefficient for the *entire* A block row?
            # No, Strassen works on the 2x2 matrix of blocks.
            # A_block_matrix is 2x2.
            # U_block[bi][r_block] is the coefficient for A_block[bi, bj]?
            # No, U_block[bi][r_block] is the coefficient for A_block[bi, bj] is not right.
            # U_block is 2x7. U_block[bi][r_block] is the coefficient for A_block[bi, bj]?
            # In Strassen, U is a 2x2 matrix of coefficients?
            # No, U is a linear combination of the 4 blocks.
            # U = sum_{bi, bj} U_coeff[bi, bj] A_block[bi, bj].
            
            # Let's redefine U_block as 2x2x7?
            # U_coeff[bi, bj, r_block].
            
    # Let's just use the known rank 49 decomposition directly if possible.
    # Or, simpler: use schoolbook for 4x4? Rank 64.
    # 64 + 16 + 4 + 4 + 1 = 89.
    # 89 < 125.
    # Let's use schoolbook for 4x4 to be safe and simple.
    
    U = [[0]*64 for _ in range(16)]
    V = [[0]*64 for _ in range(16)]
    W = [[0]*64 for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Split 5x5 into 4x4 and 1x1 parts.
    # A = [[A44, A41], [A14, A11]]
    # B = [[B44, B41], [B14, B11]]
    
    # 1. A44 * B44 (4x4 * 4x4 -> 4x4). Rank 64 (schoolbook) or 49 (Strassen).
    # Let's use schoolbook for simplicity, rank 64.
    # 2. A41 * B14 (4x1 * 1x4 -> 4x4). Rank 16.
    # 3. A41 * B11 (4x1 * 1x1 -> 4x1). Rank 4.
    # 4. A14 * B41 (1x4 * 4x1 -> 1x4). Rank 4.
    # 5. A11 * B11 (1x1 * 1x1 -> 1x1). Rank 1.
    
    total_rank = 64 + 16 + 4 + 4 + 1
    U = [[0]*total_rank for _ in range(25)]
    V = [[0]*total_rank for _ in range(25)]
    W = [[0]*total_rank for _ in range(25)]
    
    col = 0
    
    # 1. A44 * B44
    # Indices: A[0..3, 0..3], B[0..3, 0..3], C[0..3, 0..3]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*5 + i][col] = 1
                col += 1
                
    # 2. A41 * B14
    # A[0..3, 4] (col 4), B[4, 0..3] (row 4)
    # C[i, j] += A[i, 4] * B[4, j]
    for i in range(4):
        for j in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + j][col] = 1
            W[j*5 + i][col] = 1
            col += 1
            
    # 3. A41 * B11
    # A[0..3, 4], B[4, 4]
    # C[i, 4] += A[i, 4] * B[4, 4]
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*5 + i][col] = 1
        col += 1
        
    # 4. A14 * B41
    # A[4, 0..3], B[0..3, 4]
    # C[4, j] += A[4, j] * B[j, 4]
    for j in range(4):
        U[4*5 + j][col] = 1
        V[j*5 + 4][col] = 1
        W[4*5 + 4][col] = 1 # Wait. C[4, j]? No.
        # C index is k*n + i.
        # Here i=4 (row of C), k=j (col of C)?
        # A is 5x5, B is 5x5.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # Term: A[4, j] B[j, 4].
        # i=4, k=4? No.
        # A[4, j] is row 4, col j.
        # B[j, 4] is row j, col 4.
        # Result C[4, 4].
        # So all these terms add to C[4, 4].
        W[4*5 + 4][col] = 1
        col += 1
        
    # 5. A11 * B11
    # A[4, 4], B[4, 4]
    # C[4, 4] += A[4, 4] B[4, 4]
    U[4*5 + 4][col] = 1
    V[4*5 + 4][col] = 1
    W[4*5 + 4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
