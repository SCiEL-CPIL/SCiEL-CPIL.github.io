import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # We construct the rank 17 decomposition for 2x3x3 matrix multiplication.
    # The decomposition is based on:
    # 1. Strassen (rank 7) for the 2x2x2 core.
    # 2. Schoolbook (rank 4) for the 2x1x2 block (A_right * B_bottom_left).
    # 3. Schoolbook (rank 4) for the 2x2x1 block (A_left * B_top_right).
    # 4. Schoolbook (rank 2) for the 2x1x1 block (A_right * B_bottom_right).
    # Total: 7 + 4 + 4 + 2 = 17.
    
    # We use integer entries.
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices:
    # A (2x3): i in 0..1, j in 0..2. Index i*3 + j.
    #   A00=0, A01=1, A02=2
    #   A10=3, A11=4, A12=5
    # B (3x3): j in 0..2, k in 0..2. Index j*3 + k.
    #   B00=0, B01=1, B02=2
    #   B10=3, B11=4, B12=5
    #   B20=6, B21=7, B22=8
    # C (2x3) output W[k*n+i, r] => W[k*2 + i, r].
    #   C00=0, C01=1, C02=2  (k=0, i=0..2? No, C is 2x3. k is col 0..2, i is row 0..1)
    #   Wait, W[k*n+i, r]. n=2.
    #   k=0: W[0, r]=C[0,0], W[1, r]=C[1,0]
    #   k=1: W[2, r]=C[0,1], W[3, r]=C[1,1]
    #   k=2: W[4, r]=C[0,2], W[5, r]=C[1,2]
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # C indices: 0,1,2,3
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Block 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2, 5
    # B indices: 6, 7
    # C indices: 0,1,2,3
    
    # r=7: A02 B20
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 B21
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 B20
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 B21
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Block 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2,5,8
    # C indices: 4,5
    
    # r=11: A00 B02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 B12
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 B22
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 B02
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 B12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 B22
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify decomposition
    # T = sum_r U[:,r] V[:,r] W[:,r]^T
    # We can do a quick check if time permits, but it's static.
    
    if time.time() - start_time > deadline - 5:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
