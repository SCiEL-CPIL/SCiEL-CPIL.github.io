import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    
    # Total rank
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col_idx
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[i_off*m + j_off][col_idx] = 1
        U[(i_off+1)*m + (j_off+1)][col_idx] = 1
        V[j_off*p + k_off][col_idx] = 1
        V[(j_off+1)*p + (k_off+1)][col_idx] = 1
        W[k_off*n + i_off][col_idx] = 1
        W[(k_off+1)*n + (i_off+1)][col_idx] = 1
        col_idx += 1
        
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U[i_off*m + (j_off+1)][col_idx] = 1
        U[(i_off+1)*m + (j_off+1)][col_idx] = 1
        V[j_off*p + k_off][col_idx] = 1
        W[k_off*n + (i_off+1)][col_idx] = 1
        W[(k_off+1)*n + (i_off+1)][col_idx] = -1
        col_idx += 1
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[i_off*m + j_off][col_idx] = 1
        V[(j_off)*p + (k_off+1)][col_idx] = 1
        V[(j_off+1)*p + (k_off+1)][col_idx] = -1
        W[(k_off+1)*n + i_off][col_idx] = 1
        W[(k_off+1)*n + (i_off+1)][col_idx] = 1
        col_idx += 1
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i_off+1)*m + (j_off+1)][col_idx] = 1
        V[(j_off+1)*p + k_off][col_idx] = 1
        V[j_off*p + k_off][col_idx] = -1
        W[k_off*n + i_off][col_idx] = 1
        W[k_off*n + (i_off+1)][col_idx] = 1
        col_idx += 1
        
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U[i_off*m + j_off][col_idx] = 1
        U[i_off*m + (j_off+1)][col_idx] = 1
        V[(j_off+1)*p + (k_off+1)][col_idx] = 1
        W[k_off*n + i_off][col_idx] = -1
        W[(k_off+1)*n + i_off][col_idx] = 1
        col_idx += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i_off+1)*m + j_off][col_idx] = 1
        U[i_off*m + j_off][col_idx] = -1
        V[j_off*p + k_off][col_idx] = 1
        V[j_off*p + (k_off+1)][col_idx] = 1
        W[(k_off+1)*n + (i_off+1)][col_idx] = 1
        col_idx += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[i_off*m + (j_off+1)][col_idx] = 1
        U[(i_off+1)*m + (j_off+1)][col_idx] = -1
        V[(j_off+1)*p + k_off][col_idx] = 1
        V[(j_off+1)*p + (k_off+1)][col_idx] = 1
        W[k_off*n + i_off][col_idx] = 1
        col_idx += 1

    def add_schoolbook(i, j, k):
        nonlocal col_idx
        U[i*m + j][col_idx] = 1
        V[j*p + k][col_idx] = 1
        W[k*n + i][col_idx] = 1
        col_idx += 1

    # Block 1: k in {0, 1}
    add_strassen(0, 0, 0)
    
    # Schoolbook for Block 1: j=2, k in {0, 1}
    for i in [0, 1]:
        for k in [0, 1]:
            add_schoolbook(i, 2, k)
            
    # Block 2: k in {2, 3}
    add_strassen(0, 0, 2)
    
    # Schoolbook for Block 2:
    # 1. j=2, k in {2, 3, 4}
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_schoolbook(i, 2, k)
            
    # 2. j in {0, 1}, k=4
    for i in [0, 1]:
        for j in [0, 1]:
            add_schoolbook(i, j, 4)

    return {"U": U, "V": V, "W": W}
