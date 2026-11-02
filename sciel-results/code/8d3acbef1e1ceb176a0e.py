import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # M1 = A00 * (B00 + B11)
    # M2 = A10 * (B00 - B11)
    # M3 = (A00 + A01) * B11
    # M4 = (A10 - A00) * (B01 + B10)
    # M5 = (A01 + A11) * B00
    # M6 = (A00 + A10) * (B10 - B00)
    # M7 = (A01 + A11) * (B11 - B01)
    
    def add_strassen_block(i_base, j_base, k_base, start_col):
        nonlocal col
        # U indices: i_base..i_base+1, j_base..j_base+1
        # V indices: j_base..j_base+1, k_base..k_base+1
        # W indices: k_base..k_base+1, i_base..i_base+1
        
        # M1
        U[i_base * m + j_base][start_col] = 1
        V[j_base * p + k_base][start_col] = 1
        V[(j_base+1) * p + (k_base+1)][start_col] = 1
        W[k_base * n + i_base][start_col] = 1
        col += 1
        
        # M2
        U[(i_base+1) * m + j_base][start_col] = 1
        V[j_base * p + k_base][start_col] = 1
        V[(j_base+1) * p + (k_base+1)][start_col] = -1
        W[k_base * n + (i_base+1)][start_col] = 1
        col += 1
        
        # M3
        U[i_base * m + j_base][start_col] = 1
        U[i_base * m + (j_base+1)][start_col] = 1
        V[(j_base+1) * p + (k_base+1)][start_col] = 1
        W[(k_base+1) * n + i_base][start_col] = 1
        col += 1
        
        # M4
        U[(i_base+1) * m + j_base][start_col] = 1
        U[i_base * m + j_base][start_col] = -1
        V[j_base * p + (k_base+1)][start_col] = 1
        V[(j_base+1) * p + k_base][start_col] = 1
        W[k_base * n + (i_base+1)][start_col] = 1
        W[(k_base+1) * n + (i_base+1)][start_col] = 1
        col += 1
        
        # M5
        U[i_base * m + (j_base+1)][start_col] = 1
        U[(i_base+1) * m + (j_base+1)][start_col] = 1
        V[j_base * p + k_base][start_col] = 1
        W[k_base * n + i_base][start_col] = 1
        W[k_base * n + (i_base+1)][start_col] = 1
        col += 1
        
        # M6
        U[i_base * m + j_base][start_col] = 1
        U[(i_base+1) * m + j_base][start_col] = 1
        V[(j_base+1) * p + k_base][start_col] = 1
        V[j_base * p + k_base][start_col] = -1
        W[k_base * n + i_base][start_col] = 1
        W[k_base * n + (i_base+1)][start_col] = 1
        col += 1
        
        # M7
        U[i_base * m + (j_base+1)][start_col] = 1
        U[(i_base+1) * m + (j_base+1)][start_col] = 1
        V[(j_base+1) * p + (k_base+1)][start_col] = 1
        V[(j_base+1) * p + k_base][start_col] = -1
        W[(k_base+1) * n + i_base][start_col] = 1
        W[(k_base+1) * n + (i_base+1)][start_col] = 1
        col += 1
        
        return col

    def add_schoolbook_term(i, j, k, start_col):
        nonlocal col
        U[i * m + j][start_col] = 1
        V[j * p + k][start_col] = 1
        W[k * n + i][start_col] = 1
        col += 1
        return col

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    col = add_strassen_block(0, 0, 0, col)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    col = add_strassen_block(0, 2, 2, col)
    
    # Rest for i in {0,1}
    for i in [0, 1]:
        # j=4, k=4
        col = add_schoolbook_term(i, 4, 4, col)
        # j=4, k in {0,1,2,3}
        for k in range(4):
            col = add_schoolbook_term(i, 4, k, col)
        # j in {0,1,2,3}, k=4
        for j in range(4):
            col = add_schoolbook_term(i, j, 4, col)
            
    # i=2
    # j in {0,1}, k in {0,1}
    for j in range(2):
        for k in range(2):
            col = add_schoolbook_term(2, j, k, col)
    # j in {2,3}, k in {2,3}
    for j in range(2, 4):
        for k in range(2, 4):
            col = add_schoolbook_term(2, j, k, col)
    # j=4, k=4
    col = add_schoolbook_term(2, 4, 4, col)
    # j=4, k in {0,1,2,3}
    for k in range(4):
        col = add_schoolbook_term(2, 4, k, col)
    # j in {0,1,2,3}, k=4
    for j in range(4):
        col = add_schoolbook_term(2, j, 4, col)
        
    return {"U": U, "V": V, "W": W}
