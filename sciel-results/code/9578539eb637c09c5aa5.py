"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """Return a rank-98 Strassen-based decomposition for 4x4 * 4x8."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen decomposition for 2x2 * 2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[2*i+j, r], V[2*j+k, r], W[2*k+i, r]
    U_2 = [[0]*7 for _ in range(4)]
    V_2 = [[0]*7 for _ in range(4)]
    W_2 = [[0]*7 for _ in range(4)]
    
    # m1 = (A00+A11)(B00+B11)
    U_2[0][0] = 1; U_2[3][0] = 1
    V_2[0][0] = 1; V_2[3][0] = 1
    W_2[0][0] = 1; W_2[3][0] = 1
    
    # m2 = (A10+A11)B00
    U_2[2][1] = 1; U_2[3][1] = 1
    V_2[0][1] = 1
    W_2[2][1] = 1; W_2[3][1] = 1
    
    # m3 = A00(B01-B11)
    U_2[0][2] = 1
    V_2[1][2] = 1; V_2[3][2] = -1
    W_2[1][2] = 1; W_2[3][2] = 1
    
    # m4 = A11(B10-B00)
    U_2[3][3] = 1
    V_2[2][3] = 1; V_2[0][3] = -1
    W_2[2][3] = 1; W_2[0][3] = 1
    
    # m5 = (A00+A01)B11
    U_2[0][4] = 1; U_2[1][4] = 1
    V_2[3][4] = 1
    W_2[3][4] = 1; W_2[1][4] = 1
    
    # m6 = (A10-A00)(B00+B01)
    U_2[2][5] = 1; U_2[0][5] = -1
    V_2[0][5] = 1; V_2[1][5] = 1
    W_2[2][5] = 1; W_2[0][5] = 1
    
    # m7 = (A01-A11)(B10+B11)
    U_2[1][6] = 1; U_2[3][6] = -1
    V_2[2][6] = 1; V_2[3][6] = 1
    W_2[1][6] = 1; W_2[3][6] = 1
    
    # Combine two 4x4*4x4 to get 4x4*4x8
    # The 4x4*4x4 has rank 49 (7x7 Strassen)
    # We need to handle the index mapping for 4x4 blocks
    
    # For 4x4 * 4x4, split into 2x2 blocks:
    # A is 4x4, split into 2x2 blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), 
    # A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B is 4x4, split similarly
    # C is 4x4, split similarly
    
    # The 7 Strassen multiplications on 2x2 blocks give 7 sub-problems of 2x2*2x2
    # Each 2x2*2x2 has rank 7
    # Total rank for 4x4*4x4 is 7*7 = 49
    
    # For 4x4*4x8, split B and C into two 4x4 halves
    # B1 = B[:, 0:4], B2 = B[:, 4:8]
    # C1 = C[:, 0:4], C2 = C[:, 4:8]
    # Compute C1 = A*B1 and C2 = A*B2 using 4x4*4x4 Strassen
    # Total rank = 49 + 49 = 98
    
    # Build U, V, W for 4x4*4x8
    rank = 98
    U = [[0]*rank for _ in range(16)]  # 4*4
    V = [[0]*rank for _ in range(32)]  # 4*8
    W = [[0]*rank for _ in range(32)]  # 8*4
    
    # For each of the two halves (B1, B2 and C1, C2)
    for half in range(2):
        base_col = half * 49
        # For 4x4*4x4, the indices are:
        # U[4*i_block + 2*i_sub + j, col] where i_block, i_sub in {0,1}
        # V[4*j_block + 2*j_sub + k, col]
        # W[4*k_block + 2*k_sub + i, col]
        
        # The 49 rank terms come from 7x7 combinations of the 2x2 Strassen
        # Let's iterate over the 7 block-level Strassen terms and 7 sub-level terms
        
        # Block-level Strassen terms (7 of them):
        # m1_block = (A00_block + A11_block) * (B00_block + B11_block)
        # ... etc
        
        # Actually, it's easier to think of it as:
        # For each of the 7 block-level multiplications, we have a 2x2*2x2 problem
        # which has rank 7
        # So total 49 terms
        
        # Let me define the block-level Strassen coefficients
        # A_blocks: A00, A01, A10, A11 (each 2x2)
        # B_blocks: B00, B01, B10, B11 (each 2x2)
        # C_blocks: C00, C01, C10, C11 (each 2x2)
        
        # Block Strassen:
        # m1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        # m2 = (A10+A11)B00 -> contributes to C10, C11
        # m3 = A00(B01-B11) -> contributes to C01, C11
        # m4 = A11(B10-B00) -> contributes to C10, C00
        # m5 = (A00+A01)B11 -> contributes to C11, C01
        # m6 = (A10-A00)(B00+B01) -> contributes to C10, C00
        # m7 = (A01-A11)(B10+B11) -> contributes to C01, C10
        
        # For each block term, we apply 2x2 Strassen on the 2x2 matrices
        
        # Let's build it systematically
        # For block term t (0..6), we have specific A and B block combinations
        # Then for each, we apply 2x2 Strassen (7 terms)
        
        # Block term definitions (which A blocks and B blocks with signs):
        # t=0: A: [1,0,0,1], B: [1,0,0,1] -> C: [1,0,0,1]
        # t=1: A: [0,0,1,1], B: [1,0,0,0] -> C: [0,0,1,1]
        # t=2: A: [1,0,0,0], B: [0,1,0,-1] -> C: [0,1,0,1]
        # t=3: A: [0,0,0,1], B: [0,0,1,-1] -> C: [1,0,1,0]
        # t=4: A: [1,1,0,0], B: [0,0,0,1] -> C: [0,1,0,1]
        # t=5: A: [0,0,1,-1], B: [1,1,0,0] -> C: [1,0,1,0]
        # t=6: A: [0,1,0,-1], B: [0,0,1,1] -> C: [0,1,1,0]
        
        block_defs = [
            # (A_coeffs, B_coeffs, C_coeffs) where coeffs are [A00, A01, A10, A11]
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # t=0
            ([0,0,1,1], [1,0,0,0], [0,0,1,1]),  # t=1
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # t=2
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),  # t=3
            ([1,1,0,0], [0,0,0,1], [0,1,0,1]),  # t=4
            ([0,0,1,-1], [1,1,0,0], [1,0,1,0]),  # t=5
            ([0,1,0,-1], [0,0,1,1], [0,1,1,0]),  # t=6
        ]
        
        for bt, (a_coeffs, b_coeffs, c_coeffs) in enumerate(block_defs):
            # For this block term, we have a 2x2*2x2 problem
            # Apply 2x2 Strassen (7 terms)
            for st in range(7):
                col = base_col + bt * 7 + st
                
                # U coefficients for this term
                # A is a linear combination of A_blocks
                # A_block (bi, bj) where bi, bj in {0,1}
                # A_block is 2x2, with elements at positions:
                # A_block[0,0] -> global A[2*bi, 2*bj]
                # A_block[0,1] -> global A[2*bi, 2*bj+1]
                # A_block[1,0] -> global A[2*bi+1, 2*bj]
                # A_block[1,1] -> global A[2*bi+1, 2*bj+1]
                
                # For 2x2 Strassen term st, we need to know which A sub-block elements
                # are used with which coefficients
                # From U_2: U_2[2*i_sub + j_sub, st]
                # where i_sub, j_sub in {0,1}
                
                for i_sub in range(2):
                    for j_sub in range(2):
                        coeff = U_2[i_sub * 2 + j_sub][st]
                        if coeff != 0:
                            # This corresponds to A_sub[i_sub, j_sub] in the 2x2 block
                            # The block index is (bi, bj) where:
                            # bi = 0 or 1, bj = 0 or 1
                            # We need to find which block (bi, bj) this sub-element belongs to
                            # Actually, the 2x2 block is A[bi, bj] which is 2x2
                            # A_sub[i_sub, j_sub] = A[2*bi + i_sub, 2*bj + j_sub]
                            # So for a given (i_sub, j_sub), it can be in any block (bi, bj)
                            # The coefficient from the block-level term is a_coeffs[bi*2 + bj]
                            for bi in range(2):
                                for bj in range(2):
                                    block_coeff = a_coeffs[bi * 2 + bj]
                                    total_coeff = coeff * block_coeff
                                    if total_coeff != 0:
                                        row = (bi * 2 + i_sub) * 2 + (bj * 2 + j_sub)
                                        U[row][col] += total_coeff
                
                # V coefficients
                for j_sub in range(2):
                    for k_sub in range(2):
                        coeff = V_2[j_sub * 2 + k_sub][st]
                        if coeff != 0:
                            for bj in range(2):
                                for bk in range(2):
                                    block_coeff = b_coeffs[bj * 2 + bk]
                                    total_coeff = coeff * block_coeff
                                    if total_coeff != 0:
                                        # V row index for 4x4*4x4: (bj * 2 + j_sub) * 2 + (bk * 2 + k_sub)
                                        # But for 4x4*4x8, B is 4x8, so V is 4*8=32
                                        # B block (bj, bk) where bj in {0,1} (rows of A side), bk in {0,1} (cols)
                                        # Actually, B is 4x4 for this half, so B[2*bj + j_sub, 2*bk + k_sub]
                                        # V row = (2*bj + j_sub) * 4 + (2*bk + k_sub)
                                        # Wait, B is m x p = 4 x 4 for this half
                                        # V is (m*p, rank) = (4*4, rank) = (16, rank) for 4x4*4x4
                                        # But for 4x4*4x8, V is (4*8, rank) = (32, rank)
                                        # The two halves correspond to cols 0-3 and 4-7 of B
                                        # For half=0, B cols 0-3; for half=1, B cols 4-7
                                        row = (2 * bj + j_sub) * 4 + (half * 4 + 2 * bk + k_sub)
                                        V[row][col] += total_coeff
                
                # W coefficients
                for k_sub in range(2):
                    for i_sub in range(2):
                        coeff = W_2[k_sub * 2 + i_sub][st]
                        if coeff != 0:
                            for bk in range(2):
                                for bi in range(2):
                                    block_coeff = c_coeffs[bk * 2 + bi]
                                    total_coeff = coeff * block_coeff
                                    if total_coeff != 0:
                                        # W row index for 4x4*4x4: (bk * 2 + k_sub) * 2 + (bi * 2 + i_sub)
                                        # For 4x4*4x8, W is (8*4, rank) = (32, rank)
                                        # W[k*n + i, r] where k is col of B (0..7), i is row of A (0..3)
                                        # C is 4x8, so C[i, k] where i in 0..3, k in 0..7
                                        # W row = k * n + i = k * 4 + i
                                        # k = half * 4 + bk * 2 + k_sub
                                        # i = bi * 2 + i_sub
                                        k_global = half * 4 + bk * 2 + k_sub
                                        i_global = bi * 2 + i_sub
                                        row = k_global * 4 + i_global
                                        W[row][col] += total_coeff
    
    # Convert to integers if possible
    def clean_matrix(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in mat]
    
    U = clean_matrix(U)
    V = clean_matrix(V)
    W = clean_matrix(W)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
