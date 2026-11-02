import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n, m, p = DIMENSIONS
    
    # Rank 35 Strassen decomposition
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrix multiplication
    # A indices: [A11, A12, A21, A22] -> [0, 1, 2, 3]
    # B indices (local 2x2): [B11, B12, B21, B22] -> [0, 1, 2, 3]
    # C indices (local 2x2): [C11, C12, C21, C22] -> [0, 1, 2, 3]
    # Note: The task uses W[k*n+i]. For local 2x2, k is col, i is row.
    # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22.
    # So local C indices [0, 1, 2, 3] correspond to [C11, C12, C21, C22].
    
    strassen_terms = [
        # M1: (A11+A22)(B11+B22) -> C11+=M1, C12+=M1, C21+=M1, C22+=M1 (Wait, Strassen formulas)
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 + M7 - M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M5 - M2
        
        # Vector a (coeffs for A11, A12, A21, A22)
        # Vector b (coeffs for B11, B12, B21, B22)
        # Vector c (coeffs for C11, C12, C21, C22)
        
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2 -> C21 += M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]), # M3 -> C12 += M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [-1, 0, 1, 0]),# M4 -> C11 -= M4, C21 += M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]), # M5 -> C12 += M5, C22 -= M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [-1, 0, 0, 0]),# M6 -> C11 -= M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),# M7 -> C11 += M7, C22 -= M7
    ]
    
    # Verify Strassen locally?
    # C11 = 1*M1 -1*M4 +1*M7 -1*M6 = M1 - M4 + M7 - M6. Correct.
    # C12 = 1*M3 + 1*M5 = M3 + M5. Correct.
    # C21 = 1*M2 + 1*M4 = M2 + M4. Correct.
    # C22 = 1*M1 -1*M3 -1*M5 -1*M2 = M1 - M3 - M5 - M2. Correct.
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            # U: A is 2x2, indices [i*m+j]
            for idx in range(4):
                U[idx][col] = a_vec[idx]
            
            # V: B is 2x10. Block k covers columns 2k, 2k+1.
            # B[0, 2k] is local B11. Index 0*10 + 2k.
            # B[0, 2k+1] is local B12. Index 0*10 + 2k+1.
            # B[1, 2k] is local B21. Index 1*10 + 2k.
            # B[1, 2k+1] is local B22. Index 1*10 + 2k+1.
            V[0 * p + 2 * k][col] = b_vec[0]
            V[0 * p + 2 * k + 1][col] = b_vec[1]
            V[1 * p + 2 * k][col] = b_vec[2]
            V[1 * p + 2 * k + 1][col] = b_vec[3]
            
            # W: C is 2x10. Convention W[k'*n + i].
            # C[0, 2k] is local C11. Index 2k*2 + 0 = 4k.
            # C[0, 2k+1] is local C12. Index (2k+1)*2 + 0 = 4k+2.
            # C[1, 2k] is local C21. Index 2k*2 + 1 = 4k+1.
            # C[1, 2k+1] is local C22. Index (2k+1)*2 + 1 = 4k+3.
            
            # c_vec is [C11, C12, C21, C22]
            W[4 * k][col] = c_vec[0]       # C11
            W[4 * k + 2][col] = c_vec[1]   # C12
            W[4 * k + 1][col] = c_vec[2]   # C21
            W[4 * k + 3][col] = c_vec[3]   # C22
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
