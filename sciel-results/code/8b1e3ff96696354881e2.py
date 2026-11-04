"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import sys

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Shapes: (4, 7), (4, 7), (4, 7).
    Indices: i*2 + j.
    """
    U = [[0.0] * 7 for _ in range(4)]
    V = [[0.0] * 7 for _ in range(4)]
    W = [[0.0] * 7 for _ in range(4)]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1 # This is not right, W is output C
    
    # Let's use the standard Strassen coefficients directly.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices 0,1,2,3 correspond to A00, A01, A10, A11 (row-major)
    # 0: 00, 1: 01, 2: 10, 3: 11
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # C00 gets M1, C11 gets M1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1 # C10 gets M2, C11 gets -M2
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1 # C01 gets M3, C11 gets M3
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1 # C00 gets -M4, C10 gets M4
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[3][4] = 1 # C00 gets M5, C11 gets M5
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11 gets M6
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[1][6] = 1 # C00 gets -M7, C01 gets M7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 5
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: Strassen for first 4 columns (4x4x4)
    # We use 7 Strassen 2x2 blocks.
    # Each 2x2 block uses 7 scalar multiplications.
    # Total 49 terms.
    
    U_2, V_2, W_2 = _get_strassen_2x2()
    
    # Define the 7 Strassen terms for 4x4
    # Blocks:
    # A11: [0,0]-[1,1], A12: [0,2]-[1,3]
    # A21: [2,0]-[3,1], A22: [2,2]-[3,3]
    # Same for B.
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We will iterate through the 7 terms of the 4x4 Strassen.
    # For each term, we have a linear combination of 2x2 blocks.
    # We map these to the 4x4 indices.
    
    # Helper to add a 2x2 term scaled/shifted to 4x4
    def add_2x2_term(U_offset, V_offset, W_offset, sign_U, sign_V, sign_W, target_U, target_V, target_W):
        # This is too complex. Let's just generate the 49 terms explicitly.
        pass

    # Explicit Generation of 49 terms
    # Term structure: (U_coeff, V_coeff, W_coeff)
    # U is 16x1, V is 20x1, W is 20x1
    
    terms = []
    
    # Strassen 4x4 formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We expand each M into 7 scalar terms using U_2, V_2, W_2.
    
    # Block Mappings for A (4x4):
    # A11: rows 0-1, cols 0-1 -> indices 0,1,2,3
    # A12: rows 0-1, cols 2-3 -> indices 4,5,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,10,11
    # A22: rows 2-3, cols 2-3 -> indices 12,13,14,15
    
    # Block Mappings for B (4x4 for first 4 cols):
    # B11: rows 0-1, cols 0-1 -> indices 0,1,5,6 (col stride 5)
    # B12: rows 0-1, cols 2-3 -> indices 2,3,7,8
    # B21: rows 2-3, cols 0-1 -> indices 10,11,15,16
    # B22: rows 2-3, cols 2-3 -> indices 12,13,17,18
    
    # Block Mappings for C (4x4 for first 4 cols):
    # C11: rows 0-1, cols 0-1 -> indices 0,1,4,5 (col stride 4)
    # C12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # Function to add a 2x2 term to the list
    def add_term(U_map, V_map, W_map, coeff_U, coeff_V, coeff_W):
        # U_map: list of 4 indices in A
        # V_map: list of 4 indices in B
        # W_map: list of 4 indices in C
        # coeff_U, V, W: lists of 4 coeffs for the 2x2 block
        # This assumes the 2x2 term is applied to these blocks.
        pass

    # This is getting too messy. I will use a simpler, known low-rank decomposition if possible.
    # Actually, the rank 65 construction is safe.
    # I will hardcode the 49 terms by running the Strassen logic in the code.
    
    U = [[0] * 65 for _ in range(16)]
    V = [[0] * 65 for _ in range(20)]
    W = [[0] * 65 for _ in range(20)]
    
    col = 0
    
    # Strassen 4x4 part
    # We need to map the 7 Strassen 2x2 terms to the 4x4 structure.
    
    # M1 = (A11+A22)(B11+B22)
    # A11+A22: A indices 0,1,2,3 (coeff 1) and 12,13,14,15 (coeff 1)
    # B11+B22: B indices 0,1,5,6 (coeff 1) and 12,13,17,18 (coeff 1)
    # Output: C11 (0,1,4,5) and C22 (10,11,14,15) get contributions.
    
    # Instead of hardcoding, I'll use the 2x2 Strassen matrices U_2, V_2, W_2.
    
    def apply_2x2(U_2, V_2, W_2, A_idx, B_idx, C_idx, col_start):
        # A_idx, B_idx, C_idx are lists of 4 indices for the 2x2 blocks
        for t in range(7):
            for a in range(4):
                if U_2[a][t] != 0:
                    U[A_idx[a]][col_start + t] += U_2[a][t]
            for b in range(4):
                if V_2[b][t] != 0:
                    V[B_idx[b]][col_start + t] += V_2[b][t]
            for c in range(4):
                if W_2[c][t] != 0:
                    W[C_idx[c]][col_start + t] += W_2[c][t]

    # This approach is flawed because Strassen 4x4 combines 2x2 blocks linearly BEFORE multiplication.
    # The rank 49 comes from 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is rank 7.
    # So we just need to define the 7 linear combinations for A, B, and C.
    
    # Let's define the 7 terms for 4x4 Strassen:
    # Term 0: (A11+A22) * (B11+B22) -> C11, C22
    # Term 1: (A21+A22) * B11 -> C21, C22
    # Term 2: A11 * (B12-B22) -> C12, C22
    # Term 3: A22 * (B21-B11) -> C11, C21
    # Term 4: (A11+A12) * B22 -> C11, C22
    # Term 5: (A21-A11) * (B11+B12) -> C22
    # Term 6: (A12-A22) * (B21+B22) -> C11, C12
    
    # We will instantiate each of these as a 2x2x2 tensor.
    
    # A blocks:
    A11 = [0, 1, 2, 3]
    A12 = [4, 5, 6, 7]
    A21 = [8, 9, 10, 11]
    A22 = [12, 13, 14, 15]
    
    # B blocks (cols 0-3):
    B11 = [0, 1, 5, 6]
    B12 = [2, 3, 7, 8]
    B21 = [10, 11, 15, 16]
    B22 = [12, 13, 17, 18]
    
    # C blocks (cols 0-3):
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    # We need to map the 2x2 Strassen coefficients to these blocks.
    # For each of the 7 terms, we define the A, B, C blocks and signs.
    
    # Term 0: A=A11+A22, B=B11+B22, C=C11+C22
    # A11 coeff 1, A22 coeff 1
    # B11 coeff 1, B22 coeff 1
    # C11 coeff 1, C22 coeff 1
    
    # We can't just add them because the 2x2 tensor is not linear in the blocks in that way?
    # Yes it is. The tensor is multilinear.
    # So we can sum the tensors.
    
    # Let's define a function to add a scaled 2x2 Strassen tensor.
    
    def add_strassen_2x2(A_blocks, B_blocks, C_blocks, col_offset):
        # A_blocks: list of (idx, coeff)
        # B_blocks: list of (idx, coeff)
        # C_blocks: list of (idx, coeff)
        
        for t in range(7):
            for a_idx, a_coeff in A_blocks:
                for k in range(4):
                    U[a_idx[k]][col_offset + t] += U_2[k][t] * a_coeff
            for b_idx, b_coeff in B_blocks:
                for k in range(4):
                    V[b_idx[k]][col_offset + t] += V_2[k][t] * b_coeff
            for c_idx, c_coeff in C_blocks:
                for k in range(4):
                    W[c_idx[k]][col_offset + t] += W_2[k][t] * c_coeff

    # Term 0
    add_strassen_2x2([(A11, 1), (A22, 1)], [(B11, 1), (B22, 1)], [(C11, 1), (C22, 1)], 0)
    # Term 1
    add_strassen_2x2([(A21, 1), (A22, 1)], [(B11, 1)], [(C21, 1), (C22, -1)], 7)
    # Term 2
    add_strassen_2x2([(A11, 1)], [(B12, 1), (B22, -1)], [(C12, 1), (C22, 1)], 14)
    # Term 3
    add_strassen_2x2([(A22, 1)], [(B21, 1), (B11, -1)], [(C11, -1), (C21, 1)], 21)
    # Term 4
    add_strassen_2x2([(A11, 1), (A12, 1)], [(B22, 1)], [(C11, 1), (C22, 1)], 28)
    # Term 5
    add_strassen_2x2([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)], [(C22, 1)], 35)
    # Term 6
    add_strassen_2x2([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)], [(C11, -1), (C12, 1)], 42)
    
    # Part 2: Schoolbook for 5th column (col 4)
    # C[:, 4] = A * B[:, 4]
    # 16 terms.
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][49 + (i * 4 + j)] = 1
            V[j * 5 + 4][49 + (i * 4 + j)] = 1
            W[4 * 4 + i][49 + (i * 4 + j)] = 1 # W[k*n+i], k=4, i=i
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
