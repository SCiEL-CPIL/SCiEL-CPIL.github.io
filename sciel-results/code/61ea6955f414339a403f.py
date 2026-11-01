import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Attempt to find rank 6 solution (not found in literature, so we skip heavy search)
    # and fall back to Strassen rank 7.
    
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms
    # Indices for U (A matrix): 0:00, 1:01, 2:10, 3:11
    # Indices for V (B matrix): 0:00, 1:01, 2:10, 3:11
    # Indices for W (C matrix): 
    # W[k*n + i] where C[i,k]
    # C00 -> k=0, i=0 -> 0
    # C01 -> k=1, i=0 -> 2
    # C10 -> k=0, i=1 -> 1
    # C11 -> k=1, i=1 -> 3
    
    # Term 0: (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1 # C00
    W[3][0] = 1 # C11
    
    # Term 1: (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1 # C10
    W[3][1] = -1 # C11
    
    # Term 2: A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1 # C01
    W[3][2] = 1 # C11
    
    # Term 3: A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1 # C00
    W[1][3] = 1 # C10
    
    # Term 4: (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[2][4] = 1 # C01
    
    # Term 5: (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11
    
    # Term 6: (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
