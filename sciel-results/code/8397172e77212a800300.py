import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        U[i_off*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[k_off*n + i_off][col] = -1
        col += 1
        
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = -1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1

    def add_schoolbook_221(i_off, j_off, k_off):
        nonlocal col
        U[i_off*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1
        U[(i_off+1)*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1

    # Block 1: j=0,1
    add_strassen(0, 0, 0) # k=0,1
    add_strassen(0, 0, 2) # k=2,3
    add_strassen(0, 0, 4) # k=4,5
    add_schoolbook_221(0, 0, 6) # k=6
    
    # Block 2: j=2,3
    add_strassen(0, 2, 0) # k=0,1
    add_strassen(0, 2, 2) # k=2,3
    add_strassen(0, 2, 4) # k=4,5
    add_schoolbook_221(0, 2, 6) # k=6

    return {"U": U, "V": V, "W": W}
