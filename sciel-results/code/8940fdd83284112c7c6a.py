import numpy as np

def construct():
    n, m, p = 2, 3, 3
    num_outputs = p * n
    num_inputs_a = n * m
    num_inputs_b = m * p

    # Rank 17 solution based on block-parallel composition:
    # 1. Strassen decomposition for the 2x2 block of A and 2x2 block of B (indices 0,1 in m, 0,1 in p)
    # 2. Schoolbook for the remaining terms
    
    rank = 17
    U = np.zeros((num_inputs_a, rank))
    V = np.zeros((num_inputs_b, rank))
    W = np.zeros((num_outputs, rank))

    # Indices:
    # A (2x3):
    # A[0,0] -> U[0], A[0,1] -> U[1], A[0,2] -> U[2]
    # A[1,0] -> U[3], A[1,1] -> U[4], A[1,2] -> U[5]
    
    # B (3x3):
    # B[0,0] -> V[0], B[0,1] -> V[1], B[0,2] -> V[2]
    # B[1,0] -> V[3], B[1,1] -> V[4], B[1,2] -> V[5]
    # B[2,0] -> V[6], B[2,1] -> V[7], B[2,2] -> V[8]
    
    # C (2x3) Output W[k*n + i, r] = C[i,k]
    # C[0,0] -> W[0], C[0,1] -> W[1], C[0,2] -> W[2]
    # C[1,0] -> W[3], C[1,1] -> W[4], C[1,2] -> W[5]

    # --- Strassen for C[0:2, 0:2] using A[0:2, 0:2] and B[0:2, 0:2] ---
    # M1 = (A00 + A11) * (B00 + B11) -> W[0,0], W[3,0]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = A11 * (B10 - B00) -> W[1,1], W[3,1]
    U[4, 1] = 1
    V[3, 1] = 1; V[0, 1] = -1
    W[1, 1] = 1; W[3, 1] = 1
    
    # M3 = (A10 - A00) * B00 -> W[0,2], W[1,2]
    U[3, 2] = 1; U[0, 2] = -1
    V[0, 2] = 1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A00 * (B11 - B00) -> W[0,3], W[3,3]
    U[0, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00 + A01) * B10 -> W[1,4], W[3,4]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10 - A11) * (B01 + B11) -> W[0,5], W[1,5]
    U[3, 5] = 1; U[4, 5] = -1
    V[1, 5] = 1; V[4, 5] = 1
    W[0, 5] = 1; W[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11) -> W[2,6], W[4,6]
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[2, 6] = 1; W[4, 6] = 1
    
    # --- Schoolbook for the rest ---
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    # C[0,0] already handled by Strassen? No, Strassen handles C[0,0], C[0,1], C[1,0], C[1,1].
    # Wait, the Strassen part above computes:
    # W[0] (C[0,0]) gets M1 - M4 + M5 + M7? No, standard Strassen reconstruction:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M2 + M4
    # C10 = M1 - M3 - M5 - M7
    # C11 = M1 + M2 - M3 - M6
    
    # My previous assignment in the seed program was:
    # W[0,0]=1 (M1), W[3,0]=1 (M1)
    # W[1,1]=1 (M2), W[3,1]=-1 (M2? No, C11 has -M2? No C11 = M1+M2-M3-M6)
    # Let's re-verify the seed program's rank 17 assignment which was verified correct.
    
    # Seed Program Rank 17 Assignment:
    # 0: U[0,0]=1, U[4,0]=1; V[0,0]=1, V[4,0]=1; W[0,0]=1, W[3,0]=1  (M1)
    # 1: U[3,1]=1, U[4,1]=1; V[0,1]=1; W[1,1]=1, W[3,1]=-1       (M? A10+A11? No. U[3,1]=1, U[4,1]=1 is A10+A11? No, U[3]=A10, U[4]=A11. V[0]=B00. This is (A10+A11)B00. Let's call this M8)
    #    Wait, the seed program's comments say "Strassen for A[:,:2] * B[:2,:2]".
    #    Let's just copy the verified seed program's rank 17 block exactly to ensure correctness.
    
    # Re-initializing to the known good values from the seed
    U = np.zeros((num_inputs_a, rank))
    V = np.zeros((num_inputs_b, rank))
    W = np.zeros((num_outputs, rank))

    # 0
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # 1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # 2
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # 3
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # 4
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # 5
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # 6
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    # 7
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    # 8
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    # 9
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    # 10
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    # 11
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    # 12
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    # 13
    U[5, 13] = 1; V[6, 13] = 1; W[1, 13] = 1
    # 14
    U[5, 14] = 1; V[7, 14] = 1; W[3, 14] = 1
    # 15
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    # 16
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    # Verification (optional but good for safety)
    T = np.zeros((num_inputs_a, num_inputs_b, num_outputs))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                T[ia, ib, out_idx] = 1.0
                
    computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    if np.max(np.abs(computed - T)) < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 18 if verification fails (should not happen)
    rank = 18
    U = np.zeros((num_inputs_a, rank))
    V = np.zeros((num_inputs_b, rank))
    W = np.zeros((num_outputs, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                U[ia, r] = 1
                V[ib, r] = 1
                W[out_idx, r] = 1
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
