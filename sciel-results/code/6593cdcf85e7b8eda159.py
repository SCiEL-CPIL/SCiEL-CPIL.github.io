import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook block
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Helper to add Strassen block (2x2x2)
    def add_strassen(i_start, j_start, k_start):
        nonlocal col
        # i_start, j_start, k_start are the top-left indices of the 2x2x2 block
        # i: i_start, i_start+1
        # j: j_start, j_start+1
        # k: k_start, k_start+1
        
        # M1
        U[i_start * m + j_start][col] = 1
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[j_start * p + k_start][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[k_start * n + i_start][col] = 1
        W[(k_start+1) * n + (i_start+1)][col] = 1
        col += 1
        
        # M2
        U[(i_start+1) * m + j_start][col] = 1
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[j_start * p + k_start][col] = 1
        W[(k_start) * n + (i_start+1)][col] = 1
        col += 1
        
        # M3
        U[i_start * m + j_start][col] = 1
        V[j_start * p + (k_start+1)][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = -1
        W[(k_start+1) * n + i_start][col] = 1
        col += 1
        
        # M4
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + j_start][col] = 1 # Wait, V index is j*p+k.
        # Correction: V[j_start * p + k_start] etc.
        # M4: a11 (b10 - b00)
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + k_start][col] = 1
        V[j_start * p + k_start][col] = -1
        W[k_start * n + (i_start+1)][col] = 1
        col += 1
        
        # M5
        U[i_start * m + j_start][col] = 1
        U[i_start * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[(k_start+1) * n + i_start][col] = 1
        col += 1
        
        # M6
        U[(i_start+1) * m + j_start][col] = 1
        U[i_start * m + j_start][col] = -1
        V[j_start * p + k_start][col] = 1
        V[j_start * p + (k_start+1)][col] = 1
        W[k_start * n + (i_start+1)][col] = 1
        col += 1
        
        # M7
        U[i_start * m + (j_start+1)][col] = 1
        U[(i_start+1) * m + (j_start+1)][col] = -1
        V[(j_start+1) * p + k_start][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[(k_start+1) * n + i_start][col] = 1
        col += 1

    # Block 1: i=2, all j, k. Rank 28.
    add_schoolbook(2, 3, 0, 4, 0, 7)
    
    # Block 2.1: i=0,1; j=0,1; k=0,1 (Strassen) + k=2 (Schoolbook 2x2x1)
    add_strassen(0, 0, 0)
    add_schoolbook(0, 2, 0, 2, 2, 3)
    
    # Block 2.2: i=0,1; j=2,3; k=0,1 (Strassen) + k=2 (Schoolbook 2x2x1)
    add_strassen(0, 2, 0)
    add_schoolbook(0, 2, 2, 4, 2, 3)
    
    # Block 2.3: i=0,1; j=0,1; k=3,4 (Strassen) + k=5,6 (Strassen)
    add_strassen(0, 0, 3)
    add_strassen(0, 0, 5)
    
    # Block 2.4: i=0,1; j=2,3; k=3,4 (Strassen) + k=5,6 (Strassen)
    add_strassen(0, 2, 3)
    add_strassen(0, 2, 5)
    
    return {"U": U, "V": V, "W": W}
