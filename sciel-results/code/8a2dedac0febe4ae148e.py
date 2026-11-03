import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a rank-97 tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy:
    1. Split B (4x7) into B1 (4x4) and B2 (4x3).
    2. Decompose A * B1 (4x4 * 4x4) using recursive Strassen (Rank 49).
    3. Decompose A * B2 (4x4 * 4x3) using schoolbook (Rank 48).
    Total Rank = 97.
    """
    n, m, p = 4, 4, 7
    rank_4x4x4 = 49
    rank_4x4x3 = 4 * 4 * 3 # 48
    total_rank = rank_4x4x4 + rank_4x4x3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4 * 4x4 (Columns 0-3 of B) ---
    # Recursive Strassen for 4x4x4
    # Base case: 2x2x2 Strassen
    # U2, V2, W2 are 4x7 matrices (flattened 2x2 indices)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # M3 = A00(B01-B11) -> C01 += M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5 = (A00+A01)B11 -> C01 += M5, C11 -= M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = 1; W2[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = -1
    
    # Define the 7 block terms for 4x4x4
    # Each term t_b corresponds to a 2x2 block multiplication X * Y = Z
    # We need to map the 2x2 indices of X, Y, Z to the 4x4 indices of A, B, C.
    # And apply the Strassen 2x2x2 decomposition to each block multiplication.
    
    # Block definitions:
    # A blocks: A00 (0-1, 0-1), A01 (0-1, 2-3), A10 (2-3, 0-1), A11 (2-3, 2-3)
    # B blocks: B00 (0-1, 0-1), B01 (0-1, 2-3), B10 (2-3, 0-1), B11 (2-3, 2-3)
    # C blocks: C00 (0-1, 0-1), C01 (0-1, 2-3), C10 (2-3, 0-1), C11 (2-3, 2-3)
    
    # For each of the 7 Strassen terms at the block level:
    # We have coefficients for A_blocks, B_blocks, and C_blocks.
    # Let's define these as dictionaries mapping (br, bc) to coeff.
    
    a_block_coeffs = [
        {(0,0):1, (1,1):1}, # T1
        {(1,0):1, (1,1):1}, # T2
        {(0,0):1},           # T3
        {(1,1):1},           # T4
        {(0,0):1, (0,1):1},  # T5
        {(1,0):1, (0,0):-1}, # T6
        {(0,1):1, (1,1):-1}, # T7
    ]
    b_block_coeffs = [
        {(0,0):1, (1,1):1}, # T1
        {(0,0):1},           # T2
        {(0,1):1, (1,1):-1}, # T3
        {(1,0):1, (0,0):-1}, # T4
        {(1,1):1},           # T5
        {(0,0):1, (0,1):1},  # T6
        {(1,0):1, (1,1):1},  # T7
    ]
    c_block_coeffs = [
        {(0,0):1, (1,1):1}, # T1
        {(1,0):1},           # T2
        {(0,1):1},           # T3
        {(0,0):1, (1,0):1},  # T4
        {(0,1):1, (1,1):-1}, # T5
        {(1,1):1},           # T6
        {(0,0):1, (1,1):-1}, # T7
    ]
    
    # For each block term b (0..6) and each sub-term s (0..6) of the 2x2x2 Strassen:
    # Total term index t = b * 7 + s
    
    for b in range(7):
        for s in range(7):
            t = b * 7 + s
            
            # U part: A
            # The A part is a sum over blocks (br, bc) with coeff a_coeff
            # For each block, we add the 2x2 Strassen U2 values scaled by a_coeff
            for (br, bc), coeff_a in a_block_coeffs[b].items():
                # Map block (br, bc) to global indices
                # Global row = br*2 + r, Global col = bc*2 + c
                # U2 index = r*2 + c
                for r in range(2):
                    for c in range(2):
                        local_u_idx = r * 2 + c
                        global_u_idx = (br * 2 + r) * m + (bc * 2 + c)
                        U[global_u_idx][t] += U2[local_u_idx][s] * coeff_a
                        
            # V part: B
            # The B part is a sum over blocks (br, bc) with coeff b_coeff
            # Note: B is 4x4 here. Global row = br*2 + r, Global col = bc*2 + c
            for (br, bc), coeff_b in b_block_coeffs[b].items():
                for r in range(2):
                    for c in range(2):
                        local_v_idx = r * 2 + c
                        global_v_idx = (br * 2 + r) * p + (bc * 2 + c) # p=4 for this part
                        V[global_v_idx][t] += V2[local_v_idx][s] * coeff_b
                        
            # W part: C
            # The C part is a sum over blocks (br, bc) with coeff c_coeff
            # C is 4x4. Global row = br*2 + r, Global col = bc*2 + c
            # W index convention: W[k*n+i, r] -> W[col*n + row, r] ? 
            # Task says: W[k*n+i, r] where k is col, i is row? 
            # "output factor follows the official trace convention W[k*n+i,r]"
            # Usually C[i,j] = sum_k A[i,k] B[k,j].
            # If W is indexed by [col*n + row, rank], then W_idx = j*n + i.
            for (br, bc), coeff_c in c_block_coeffs[b].items():
                for r in range(2):
                    for c in range(2):
                        local_w_idx = r * 2 + c
                        # Global row i = br*2 + r, Global col j = bc*2 + c
                        global_w_idx = (bc * 2 + c) * n + (br * 2 + r)
                        W[global_w_idx][t] += W2[local_w_idx][s] * coeff_c

    # --- Part 2: 4x4 * 4x3 (Columns 4-6 of B) ---
    # Schoolbook for 4x4 * 4x3
    # Rank = 4 * 4 * 3 = 48
    base_rank = 49
    for i in range(n):
        for j in range(m):
            for k in range(3): # Columns 4, 5, 6 of B
                t = base_rank + i * m + j * 3 + k
                # U: A[i, j]
                u_idx = i * m + j
                U[u_idx][t] = 1
                
                # V: B[j, 4+k]
                v_idx = j * p + (4 + k)
                V[v_idx][t] = 1
                
                # W: C[i, 4+k]
                # W index: col*n + row = (4+k)*n + i
                w_idx = (4 + k) * n + i
                W[w_idx][t] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
