"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block 1: k in {0, 1}
    # Strassen for core 2x2x2
    # Term 0: M1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[7, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1
    
    # Term 1: M2
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # Term 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[8, 2] = -1
    W[3, 2] = 1
    
    # Term 3: M4
    U[4, 3] = 1
    V[6, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1
    
    # Term 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[7, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    
    # Term 5: M6
    U[1, 5] = 1; U[3, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[4, 5] = 1
    
    # Term 6: M7
    U[3, 6] = 1; U[0, 6] = -1
    V[7, 6] = 1; V[6, 6] = 1
    W[0, 6] = 1

    # Extra terms Block 1
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(2):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Block 2: k in {2, 3}
    # Term 7: M1'
    U[0, 7] = 1; U[4, 7] = 1
    V[2, 7] = 1; V[9, 7] = 1
    W[6, 7] = 1; W[10, 7] = 1
    
    # Term 8: M2'
    U[3, 8] = 1; U[4, 8] = 1
    V[2, 8] = 1
    W[7, 8] = 1
    
    # Term 9: M3'
    U[0, 9] = 1
    V[3, 9] = 1; V[10, 9] = -1
    W[9, 9] = 1
    
    # Term 10: M4'
    U[4, 10] = 1
    V[8, 10] = 1; V[2, 10] = -1
    W[7, 10] = 1
    
    # Term 11: M5'
    U[0, 11] = 1; U[1, 11] = 1
    V[9, 11] = 1
    W[6, 11] = -1; W[9, 11] = 1
    
    # Term 12: M6'
    U[1, 12] = 1; U[3, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[10, 12] = 1
    
    # Term 13: M7'
    U[3, 13] = 1; U[0, 13] = -1
    V[9, 13] = 1; V[8, 13] = 1
    W[6, 13] = 1

    # Extra terms Block 2
    col = 14
    for i in range(3):
        for j in range(3):
            for k in range(2, 4):
                if i < 2 and j < 2:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
