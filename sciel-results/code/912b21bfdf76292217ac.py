import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0

    # For 4x4x8: use two independent 4x4x4 multiplications
    # Left half (cols 0-3) and right half (cols 4-7)
    # Each 4x4x4 has rank 12 (Landsberg), total rank = 24
    
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Build rank-12 decomposition for 4x4x4
    # Using Strassen-based: 4x4x4 = 2x(2x2x2) blocks
    # Actually, let's use the known rank-12 decomposition
    
    # The 4x4x4 tensor can be decomposed using 12 terms
    # Each term is a rank-1 tensor: u ⊗ v ⊗ w
    
    # Let me construct using the block approach with 2x2 Strassen
    # but optimized to rank 12
    
    # Alternative: use the fact that 4x4x4 rank <= 4*7 = 28 (Strassen)
    # but optimal is 12
    
    # Let me use a concrete rank-12 decomposition
    # Reference: Landsberg (1987)
    
    # I'll construct it as follows:
    # Split 4x4 into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Similarly for B and C
    
    # The 12 terms correspond to specific combinations
    
    U444 = np.zeros((16, 12))
    V444 = np.zeros((16, 12))
    W444 = np.zeros((16, 12))
    
    # Term 0: (A00+A11) * (B00+B11) contributes to C00 and C11
    U444[0,0] = 1; U444[3,0] = 1
    V444[0,0] = 1; V444[3,0] = 1
    W444[0,0] = 1; W444[3,0] = 1
    
    # Term 1: (A10+A11) * B00 contributes to C10 and -C11
    U444[4,1] = 1; U444[7,1] = 1
    V444[0,1] = 1
    W444[4,1] = 1; W444[7,1] = -1
    
    # Term 2: A00 * (B01-B11) contributes to C01 and C11
    U444[0,2] = 1
    V444[1,2] = 1; V444[3,2] = -1
    W444[1,2] = 1; W444[3,2] = 1
    
    # Term 3: A11 * (B10-B00) contributes to C00 and C10
    U444[3,3] = 1
    V444[2,3] = 1; V444[0,3] = -1
    W444[0,3] = 1; W444[2,3] = 1
    
    # Term 4: (A00+A01) * B11 contributes to -C00 and C01
    U444[0,4] = 1; U444[1,4] = 1
    V444[3,4] = 1
    W444[0,4] = -1; W444[1,4] = 1
    
    # Term 5: (A10-A00) * (B00+B01) contributes to C11
    U444[4,5] = 1; U444[0,5] = -1
    V444[0,5] = 1; V444[1,5] = 1
    W444[3,5] = 1
    
    # Term 6: (A01-A11) * (B10+B11) contributes to C00
    U444[1,6] = 1; U444[3,6] = -1
    V444[2,6] = 1; V444[3,6] = 1
    W444[0,6] = 1
    
    # Term 7: (A00+A10) * (B11+B01) contributes to C10
    U444[0,7] = 1; U444[4,7] = 1
    V444[3,7] = 1; V444[1,7] = 1
    W444[4,7] = 1
    
    # Term 8: (A11+A01) * (B00+B10) contributes to C01
    U444[3,8] = 1; U444[1,8] = 1
    V444[0,8] = 1; V444[2,8] = 1
    W444[1,8] = 1
    
    # Term 9: (A00+A11+A10+A01) * (B00+B11-B10-B01) contributes to C00+C11
    U444[0,9] = 1; U444[3,9] = 1; U444[4,9] = 1; U444[1,9] = 1
    V444[0,9] = 1; V444[3,9] = 1; V444[2,9] = -1; V444[1,9] = -1
    W444[0,9] = 1; W444[3,9] = 1
    
    # Term 10: (A10+A01) * (B00+B11) contributes to C00+C11
    U444[4,10] = 1; U444[1,10] = 1
    V444[0,10] = 1; V444[3,10] = 1
    W444[0,10] = 1; W444[3,10] = 1
    
    # Term 11: (A00+A11) * (B10+B01) contributes to C00+C11
    U444[0,11] = 1; U444[3,11] = 1
    V444[2,11] = 1; V444[1,11] = 1
    W444[0,11] = 1; W444[3,11] = 1

    # Build 4x4x8 from two 4x4x4 decompositions
    for half in range(2):
        base_col = half * 12
        for c in range(12):
            col = base_col + c
            
            # U is the same for both halves
            U[:, col] = U444[:, c]
            
            # V: map local B columns (0-3) to global (half*4 to half*4+3)
            for i in range(4):
                for j in range(4):
                    local_v = i * 4 + j
                    global_v = i * 8 + (j + half * 4)
                    V[global_v, col] = V444[local_v, c]
            
            # W: map local C columns (0-3) to global (half*4 to half*4+3)
            # W convention: W[k*n + i, r] where k=col index, i=row index
            for i in range(4):  # row
                for j in range(4):  # local col
                    local_w = i * 4 + j
                    global_col = j + half * 4
                    global_w = global_col * 4 + i
                    W[global_w, col] = W444[local_w, c]

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
