import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # We will construct a rank 28 decomposition
    # by applying Strassen's algorithm to two 2x2x2 blocks.
    # Block 1: j in {0, 1}, k in {0, 1}
    # Block 2: j in {0, 1}, k in {2, 3}
    
    # The remaining terms are schoolbook terms for:
    # j in {0, 1}, k = 4
    # j = 2, k in {0, 1, 2, 3, 4}
    
    # Total rank: 7 (Strassen 1) + 7 (Strassen 2) + 4 (j<2, k=4) + 10 (j=2, k<5) = 28
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(j_start, k_start):
        nonlocal col
        # Local indices i in {0,1}, j in {0,1}, k in {0,1}
        # Global indices:
        # U index: i*m + (j_start + j)
        # V index: (j_start + j)*p + (k_start + k)
        # W index: (k_start + k)*n + i
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11 -> idx 0, 4 (relative to m=3)
        # V: B00, B11 -> idx 0, 6 (relative to p=5)
        # W: C00, C11 -> idx 0, 3 (relative to n=2)
        
        # Note: The indices in U, V, W are global.
        
        # M1
        U[0 * m + (j_start + 0)][col] = 1
        U[1 * m + (j_start + 1)][col] = 1
        V[(j_start + 0) * p + (k_start + 0)][col] = 1
        V[(j_start + 1) * p + (k_start + 1)][col] = 1
        W[(k_start + 0) * n + 0][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        # U: A10, A11
        # V: B00
        # W: C10
        U[1 * m + (j_start + 0)][col] = 1
        U[1 * m + (j_start + 1)][col] = 1
        V[(j_start + 0) * p + (k_start + 0)][col] = 1
        W[(k_start + 0) * n + 1][col] = 1
        col += 1
        
        # M3 = A00(B10 + B11)
        # U: A00
        # V: B10, B11
        # W: C01
        U[0 * m + (j_start + 0)][col] = 1
        V[(j_start + 1) * p + (k_start + 0)][col] = 1
        V[(j_start + 1) * p + (k_start + 1)][col] = 1
        W[(k_start + 1) * n + 0][col] = 1
        col += 1
        
        # M4 = A11(B01 - B00)
        # U: A11
        # V: B01, -B00
        # W: C00, C10
        U[1 * m + (j_start + 1)][col] = 1
        V[(j_start + 0) * p + (k_start + 1)][col] = 1
        V[(j_start + 0) * p + (k_start + 0)][col] = -1
        W[(k_start + 0) * n + 0][col] = 1
        W[(k_start + 0) * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        # U: A00, A01
        # V: B11
        # W: C00, C01
        U[0 * m + (j_start + 0)][col] = 1
        U[0 * m + (j_start + 1)][col] = 1
        V[(j_start + 1) * p + (k_start + 1)][col] = 1
        W[(k_start + 0) * n + 0][col] = 1
        W[(k_start + 1) * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # U: A10, -A00
        # V: B00, B01
        # W: C11
        U[1 * m + (j_start + 0)][col] = 1
        U[0 * m + (j_start + 0)][col] = -1
        V[(j_start + 0) * p + (k_start + 0)][col] = 1
        V[(j_start + 0) * p + (k_start + 1)][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # U: A01, -A11
        # V: B10, B11
        # W: C00, C11
        U[0 * m + (j_start + 1)][col] = 1
        U[1 * m + (j_start + 1)][col] = -1
        V[(j_start + 1) * p + (k_start + 0)][col] = 1
        V[(j_start + 1) * p + (k_start + 1)][col] = 1
        W[(k_start + 0) * n + 0][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1

    # Add Block 1: j=0,1; k=0,1
    add_strassen_block(0, 0)
    
    # Add Block 2: j=0,1; k=2,3
    add_strassen_block(0, 2)
    
    # Add remaining schoolbook terms
    # j in {0, 1}, k = 4
    for j in [0, 1]:
        for i in [0, 1]:
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # j = 2, k in {0, 1, 2, 3, 4}
    for k in range(5):
        for i in [0, 1]:
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
