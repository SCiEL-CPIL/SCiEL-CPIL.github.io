import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 using Recursive Strassen (Rank 49)
    # Base case 2x2x2 Strassen
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # Reconstruction Coefficients for 2x2 blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_C = np.zeros((4, 7)) # 4 blocks in C, 7 terms
    c_C[0, 0] = 1; c_C[0, 3] = 1; c_C[0, 4] = -1; c_C[0, 6] = 1
    c_C[1, 2] = 1; c_C[1, 4] = 1
    c_C[2, 1] = 1; c_C[2, 3] = 1
    c_C[3, 0] = 1; c_C[3, 1] = -1; c_C[3, 2] = 1; c_C[3, 5] = 1
    
    # Coefficients for A and B blocks in the 7 terms
    # A00, A01, A10, A11
    c_A = np.zeros((7, 4))
    c_A[0, 0] = 1; c_A[0, 3] = 1 # M1: A00+A11
    c_A[1, 2] = 1; c_A[1, 3] = 1 # M2: A10+A11
    c_A[2, 0] = 1                 # M3: A00
    c_A[3, 3] = 1                 # M4: A11
    c_A[4, 0] = 1; c_A[4, 1] = 1 # M5: A00+A01
    c_A[5, 2] = 1; c_A[5, 0] = -1 # M6: A10-A00
    c_A[6, 1] = 1; c_A[6, 3] = -1 # M7: A01-A11
    
    c_B = np.zeros((7, 4))
    c_B[0, 0] = 1; c_B[0, 3] = 1 # M1: B00+B11
    c_B[1, 0] = 1                 # M2: B00
    c_B[2, 1] = 1; c_B[2, 3] = -1 # M3: B01-B11
    c_B[3, 2] = 1; c_B[3, 0] = -1 # M4: B10-B00
    c_B[4, 3] = 1                 # M5: B11
    c_B[5, 0] = 1; c_B[5, 1] = 1 # M6: B00+B01
    c_B[6, 2] = 1; c_B[6, 3] = 1 # M7: B10+B11

    # Build 4x4x4 decomposition
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))
    
    for k in range(7):
        for j in range(7):
            col = k * 7 + j
            # U part
            for i_block in range(2):
                for j_block in range(2):
                    coeff_a = c_A[k, i_block * 2 + j_block]
                    if coeff_a != 0:
                        for r in range(2):
                            for c in range(2):
                                I = i_block * 2 + r
                                J = j_block * 2 + c
                                U_444[I * 4 + J, col] += coeff_a * U2[r * 2 + c, j]
            # V part
            for i_block in range(2):
                for j_block in range(2):
                    coeff_b = c_B[k, i_block * 2 + j_block]
                    if coeff_b != 0:
                        for r in range(2):
                            for c in range(2):
                                I = i_block * 2 + r
                                J = j_block * 2 + c
                                V_444[I * 4 + J, col] += coeff_b * V2[r * 2 + c, j]
            # W part
            for i_block in range(2):
                for j_block in range(2):
                    coeff_c = c_C[i_block * 2 + j_block, k]
                    if coeff_c != 0:
                        for r in range(2):
                            for c in range(2):
                                I = i_block * 2 + r
                                J = j_block * 2 + c
                                W_444[I * 4 + J, col] += coeff_c * W2[r * 2 + c, j]

    # Part 2: 4x4x1 Schoolbook (Rank 16)
    rank_441 = 16
    U_441 = np.zeros((16, rank_441))
    V_441 = np.zeros((4, rank_441)) # B is 4x1, so 4 elements
    W_441 = np.zeros((4, rank_441)) # C is 4x1, so 4 elements
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i * 4 + j, col] = 1
            V_441[j, col] = 1
            W_441[i, col] = 1
            col += 1

    # Combine
    total_rank = rank_444 + rank_441
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank)) # B is 4x5 = 20
    W = np.zeros((20, total_rank)) # C is 4x5 = 20
    
    # Fill U
    U[:, :rank_444] = U_444
    U[:, rank_444:] = U_441
    
    # Fill V
    # V_444 corresponds to B[0:4, 0:4] -> indices 0..15
    V[0:16, :rank_444] = V_444
    # V_441 corresponds to B[0:4, 4] -> indices 16..19
    V[16:20, rank_444:] = V_441
    
    # Fill W
    # W_444 corresponds to C[0:4, 0:4] -> indices 0..15
    W[0:16, :rank_444] = W_444
    # W_441 corresponds to C[0:4, 4] -> indices 16..19
    W[16:20, rank_444:] = W_441

    return {"U": U, "V": V, "W": W}
