import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Component 1: 2x1x6
    # j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Strassen helper
    def add_strassen_block(j_start, k_start, base_col):
        nonlocal col
        # Local indices mapping
        # A[0,0] -> i=0, j=j_start
        # A[0,1] -> i=0, j=j_start+1
        # A[1,0] -> i=1, j=j_start
        # A[1,1] -> i=1, j=j_start+1
        # B[0,0] -> j=j_start, k=k_start
        # B[0,1] -> j=j_start, k=k_start+1
        # B[1,0] -> j=j_start+1, k=k_start
        # B[1,1] -> j=j_start+1, k=k_start+1
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        # U: A00(1), A11(1)
        U[0 * m + j_start][base_col + 0] += 1
        U[1 * m + (j_start + 1)][base_col + 0] += 1
        # V: B00(1), B11(1)
        V[j_start * p + k_start][base_col + 0] += 1
        V[(j_start + 1) * p + (k_start + 1)][base_col + 0] += 1
        # W: C00(1), C11(1)
        W[k_start * n + 0][base_col + 0] += 1
        W[(k_start + 1) * n + 1][base_col + 0] += 1
        
        # Term 2: M2 = (A10 + A11)B00
        U[1 * m + j_start][base_col + 1] += 1
        U[1 * m + (j_start + 1)][base_col + 1] += 1
        V[j_start * p + k_start][base_col + 1] += 1
        W[k_start * n + 1][base_col + 1] += 1
        W[(k_start + 1) * n + 1][base_col + 1] += -1
        
        # Term 3: M3 = A00(B01 - B11)
        U[0 * m + j_start][base_col + 2] += 1
        V[j_start * p + (k_start + 1)][base_col + 2] += 1
        V[(j_start + 1) * p + (k_start + 1)][base_col + 2] += -1
        W[(k_start + 1) * n + 0][base_col + 2] += 1
        W[(k_start + 1) * n + 1][base_col + 2] += 1
        
        # Term 4: M4 = A11(B10 - B00)
        U[1 * m + (j_start + 1)][base_col + 3] += 1
        V[(j_start + 1) * p + k_start][base_col + 3] += 1
        V[j_start * p + k_start][base_col + 3] += -1
        W[k_start * n + 0][base_col + 3] += 1
        W[k_start * n + 1][base_col + 3] += 1
        
        # Term 5: M5 = (A00 + A01)B11
        U[0 * m + j_start][base_col + 4] += 1
        U[0 * m + (j_start + 1)][base_col + 4] += 1
        V[(j_start + 1) * p + (k_start + 1)][base_col + 4] += 1
        W[k_start * n + 0][base_col + 4] += -1
        W[(k_start + 1) * n + 0][base_col + 4] += 1
        
        # Term 6: M6 = (A10 - A00)(B00 + B01)
        U[1 * m + j_start][base_col + 5] += 1
        U[0 * m + j_start][base_col + 5] += -1
        V[j_start * p + k_start][base_col + 5] += 1
        V[j_start * p + (k_start + 1)][base_col + 5] += 1
        W[(k_start + 1) * n + 1][base_col + 5] += 1
        
        # Term 7: M7 = (A01 - A11)(B10 + B11)
        U[0 * m + (j_start + 1)][base_col + 6] += 1
        U[1 * m + (j_start + 1)][base_col + 6] += -1
        V[(j_start + 1) * p + k_start][base_col + 6] += 1
        V[(j_start + 1) * p + (k_start + 1)][base_col + 6] += 1
        W[k_start * n + 0][base_col + 6] += 1
        
        return base_col + 7

    # Component 2: 2x2x6 for j in {1, 2}
    j_start = 1
    for q in range(3):
        k_start = 2 * q
        col = add_strassen_block(j_start, k_start, col)
        
    # Component 3: 2x2x6 for j in {3, 4}
    j_start = 3
    for q in range(3):
        k_start = 2 * q
        col = add_strassen_block(j_start, k_start, col)
        
    return {"U": U, "V": V, "W": W}
