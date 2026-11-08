import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by combining smaller blocks.
    # Structure:
    # A (3x4) = [A1, A2] (3x2 each)
    # B (4x6) = [B1; B2] (2x6 each)
    # C = A1 B1 + A2 B2
    
    # For each Ai Bi (3x2 by 2x6):
    # Split B_i into [B_i1, B_i2] (2x3 each)
    # A_i B_i = A_i B_i1 + A_i B_i2
    
    # For each A_i B_ij (3x2 by 2x3):
    # Use rank 17 decomposition.
    
    # Total rank = 2 (i) * 2 (j) * 17 = 68.
    
    # We need to explicitly construct the U, V, W matrices.
    # It is easier to define the small decompositions as lists of (u, v, w) tuples
    # or as matrices and then concatenate.
    
    # Let's define a helper to create a block decomposition
    # A block is defined by n_sub, m_sub, p_sub and its U, V, W matrices.
    
    # 1. 1x2 by 2x2 (Rank 4)
    # A (1x2), B (2x2) -> C (1x2)
    # C_1 = A_1 B_11 + A_2 B_21
    # C_2 = A_1 B_12 + A_2 B_22
    # This is just 4 multiplications.
    # U (1*2=2 rows, 4 cols), V (2*2=4 rows, 4 cols), W (2*1=2 rows, 4 cols)
    # Indices:
    # A: (0,0), (0,1) -> row 0, 1 in U
    # B: (0,0), (0,1), (1,0), (1,1) -> row 0,1,2,3 in V
    # C: (0,0), (0,1) -> row 0,1 in W (k*n+i, n=1, p=2? No, W is p*n. p=2, n=1. Rows 0,1)
    
    U_122 = np.zeros((2, 4))
    V_122 = np.zeros((4, 4))
    W_122 = np.zeros((2, 4))
    
    # Mult 1: A_1 * B_11 -> C_1
    U_122[0, 0] = 1 # A_1
    V_122[0, 0] = 1 # B_11
    W_122[0, 0] = 1 # C_1 (k=0, i=0)
    
    # Mult 2: A_2 * B_21 -> C_1
    U_122[1, 1] = 1 # A_2
    V_122[2, 1] = 1 # B_21 (row 1, col 0 in B -> index 1*2+0=2)
    W_122[0, 1] = 1 # C_1
    
    # Mult 3: A_1 * B_12 -> C_2
    U_122[0, 2] = 1 # A_1
    V_122[1, 2] = 1 # B_12 (row 0, col 1 in B -> index 0*2+1=1)
    W_122[1, 2] = 1 # C_2 (k=1, i=0)
    
    # Mult 4: A_2 * B_22 -> C_2
    U_122[1, 3] = 1 # A_2
    V_122[3, 3] = 1 # B_22 (row 1, col 1 in B -> index 1*2+1=3)
    W_122[1, 3] = 1 # C_2
    
    # 2. 2x2 by 2x2 (Rank 7, Strassen)
    # Standard Strassen
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # A: 0,1,2,3
    # B: 0,1,2,3
    # C: 0,1,2,3
    
    # M1 = (A11+A22)(B11+B22)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[1, 0] = 1; W_222[2, 0] = 1; W_222[3, 0] = 1
    
    # M2 = (A21+A22)A11
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = 1
    
    # M3 = A12(B11-B21)
    U_222[1, 2] = 1
    V_222[0, 2] = 1; V_222[2, 2] = -1
    W_222[0, 2] = 1; W_222[1, 2] = 1
    
    # M4 = A21(B22-B12)
    U_222[2, 3] = 1
    V_222[3, 3] = 1; V_222[1, 3] = -1
    W_222[2, 3] = 1; W_222[3, 3] = 1
    
    # M5 = (A11+A12)B22
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = 1; W_222[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[0, 5] = 1; W_222[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[1, 6] = 1; W_222[2, 6] = 1
    
    # 3. 3x2 by 2x1 (Rank 6)
    # A (3x2), B (2x1) -> C (3x1)
    # C_i = A_i1 B_1 + A_i2 B_2
    U_321 = np.zeros((6, 6))
    V_321 = np.zeros((2, 6))
    W_321 = np.zeros((3, 6))
    
    for i in range(3):
        # A_i1 * B_1
        U_321[i*2, i] = 1
        V_321[0, i] = 1
        W_321[i, i] = 1
        # A_i2 * B_2
        U_321[i*2+1, i+3] = 1
        V_321[1, i+3] = 1
        W_321[i, i+3] = 1

    # Now assemble 3x2 by 2x3 (Rank 17)
    # Split p=3 into 2+1.
    # C = C_2 + C_1
    # C_2 from 3x2 by 2x2 (Rank 11)
    # C_1 from 3x2 by 2x1 (Rank 6)
    
    U_323 = np.zeros((6, 17))
    V_323 = np.zeros((6, 17))
    W_323 = np.zeros((9, 17))
    
    # Part 1: 3x2 by 2x2
    # A is 3x2. U is 6x11.
    # Split n=3 into 1+2.
    # A_top (1x2), A_bot (2x2).
    # B (2x2).
    
    # Top part (1x2 by 2x2, Rank 4)
    # U_top (2x4), V_top (4x4), W_top (2x4)
    # Map to global indices.
    # A indices: 0,1 (top), 2,3,4,5 (bot)
    # B indices: 0,1,2,3
    # C indices: 0,1 (top rows), 2,3,4,5,6,7,8 (bot rows? No, C is 3x2. Rows 0,1,2. Cols 0,1.)
    # W shape 9x4. Rows k*n+i. n=3, p=2.
    # k=0: i=0,1,2 -> 0,1,2
    # k=1: i=0,1,2 -> 3,4,5
    
    # We need to map the local 1x2 by 2x2 output (1x2) to global 3x2 output.
    # Local C row 0 -> Global C row 0 (k=0, i=0) and row 3 (k=1, i=0)?
    # No, local n=1. Global n=3.
    # The top row of A (index 0) corresponds to global row 0.
    # So local C[0,0] -> Global C[0,0] (k=0, i=0) and C[3,0] (k=1, i=0)?
    # Wait, B is 2x2. k=0,1.
    # Local W is 2x4.
    # Local W[0, c] corresponds to k=0, i=0. Global index 0*3+0=0.
    # Local W[1, c] corresponds to k=1, i=0. Global index 1*3+0=3.
    
    # So for the top part:
    # U_323[0:2, 0:4] = U_122
    # V_323[0:4, 0:4] = V_122
    # W_323[0, 0:4] = W_122[0]
    # W_323[3, 0:4] = W_122[1]
    
    U_323[0:2, 0:4] = U_122
    V_323[0:4, 0:4] = V_122
    W_323[0, 0:4] = W_122[0]
    W_323[3, 0:4] = W_122[1]
    
    # Bot part (2x2 by 2x2, Rank 7)
    # A indices: 2,3,4,5
    # B indices: 0,1,2,3
    # C indices: rows 1,2. k=0,1.
    # Global C:
    # k=0, i=1 -> 1
    # k=0, i=2 -> 2
    # k=1, i=1 -> 4
    # k=1, i=2 -> 5
    
    # Local U_222 (4x7).
    # Local V_222 (4x7).
    # Local W_222 (4x7).
    
    U_323[2:6, 4:11] = U_222
    V_323[0:4, 4:11] = V_222
    # Map W
    # Local 0 (k=0,i=0) -> Global 1
    # Local 1 (k=0,i=1) -> Global 2
    # Local 2 (k=1,i=0) -> Global 4
    # Local 3 (k=1,i=1) -> Global 5
    W_323[1, 4:11] = W_222[0]
    W_323[2, 4:11] = W_222[1]
    W_323[4, 4:11] = W_222[2]
    W_323[5, 4:11] = W_222[3]
    
    # Part 2: 3x2 by 2x1 (Rank 6)
    # B is 2x1. Indices 4,5 in global B?
    # In the 3x2 by 2x3 block, B is 2x3.
    # We split p=3 into 2+1.
    # B_left (2x2) uses indices 0,1,2,3.
    # B_right (2x1) uses indices 4,5.
    # So for this part, V should use rows 4,5.
    
    U_323[0:6, 11:17] = U_321
    V_323[4:6, 11:17] = V_321
    # W mapping:
    # Local W_321 is 3x6.
    # Local row i -> Global row i (k=0? No, this is the p=1 part, so k=2? No, k index in W is p*n.
    # In 3x2 by 2x3, p=3.
    # W rows are k*n+i.
    # k=0: 0,1,2
    # k=1: 3,4,5
    # k=2: 6,7,8
    # The 3x2 by 2x1 part corresponds to k=2 (the last column).
    # So Global rows 6,7,8.
    W_323[6:9, 11:17] = W_321

    # Now assemble 3x4 by 4x6
    # U total: 12 rows (3*4), 68 cols
    # V total: 24 rows (4*6), 68 cols
    # W total: 18 rows (6*3), 68 cols
    
    U_final = np.zeros((12, 68))
    V_final = np.zeros((24, 68))
    W_final = np.zeros((18, 68))
    
    # 4 blocks:
    # i=0 (A top 3x2), j=0 (B left 2x3)
    # i=0, j=1 (B right 2x3)
    # i=1 (A bot 3x2), j=0
    # i=1, j=1
    
    # A indices:
    # i=0: rows 0,1,2,3,4,5 (3x2)
    # i=1: rows 6,7,8,9,10,11 (3x2)
    
    # B indices:
    # j=0: rows 0,1,2,3,4,5 (2x3)
    # j=1: rows 6,7,8,9,10,11 (2x3)
    
    # W indices:
    # j=0: k=0,1 -> rows 0..5 (3*2)
    # j=1: k=2,3,4,5 -> rows 6..17 (3*4)
    # Wait, p=6.
    # W rows k*n+i.
    # k=0: 0,1,2
    # k=1: 3,4,5
    # k=2: 6,7,8
    # k=3: 9,10,11
    # k=4: 12,13,14
    # k=5: 15,16,17
    
    # Block (0,0): A[0:6], B[0:6], W[0:6]
    U_final[0:6, 0:17] = U_323
    V_final[0:6, 0:17] = V_323
    W_final[0:6, 0:17] = W_323
    
    # Block (0,1): A[0:6], B[6:12], W[6:12]
    U_final[0:6, 17:34] = U_323
    V_final[6:12, 17:34] = V_323
    W_final[6:12, 17:34] = W_323
    
    # Block (1,0): A[6:12], B[0:6], W[0:6]
    U_final[6:12, 34:51] = U_323
    V_final[0:6, 34:51] = V_323
    W_final[0:6, 34:51] = W_323
    
    # Block (1,1): A[6:12], B[6:12], W[6:12]
    U_final[6:12, 51:68] = U_323
    V_final[6:12, 51:68] = V_323
    W_final[6:12, 51:68] = W_323
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    return construct()
# EVOLVE-BLOCK-END
