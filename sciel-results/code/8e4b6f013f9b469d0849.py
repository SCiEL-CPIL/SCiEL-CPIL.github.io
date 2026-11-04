"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import block_diag

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank-35 decomposition by splitting the 3x3x4 multiplication
    # into a 2x2x2 core (using Strassen, rank 7), a 1x1x2 edge (rank 2),
    # and the remaining cross terms (using schoolbook, rank 26).
    # Total rank = 7 + 2 + 26 = 35.
    
    rank = 35
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        
    col = 0
    # Strassen for 2x2x2 core (i in 0,1; j in 0,1; k in 0,1)
    # Indices in U: i*3+j -> (0, 1, 3, 4)
    # Indices in V: j*4+k -> (0, 1, 4, 5)
    # Indices in W: k*3+i -> (0, 1, 3, 4)
    
    # M1 = (A11+A22)(B11+B22)
    add_term(0, 0, 0, 1) # A11 B11
    add_term(0, 1, 1, 1) # A11 B22
    add_term(3, 0, 3, 1) # A22 B11
    add_term(3, 1, 4, 1) # A22 B22
    col += 1
    
    # M2 = (A21+A22)B11
    add_term(3, 0, 3, 1) # A22 B11
    add_term(1, 0, 3, 1) # A21 B11
    col += 1
    
    # M3 = A11(B21-B11)
    add_term(0, 4, 1, 1) # A11 B21
    add_term(0, 0, 1, -1) # A11 B11
    col += 1
    
    # M4 = A22(B12-B22)
    add_term(3, 1, 3, 1) # A22 B12
    add_term(3, 5, 4, -1) # A22 B22
    col += 1
    
    # M5 = (A11+A12)B22
    add_term(0, 5, 0, 1) # A11 B22
    add_term(1, 5, 0, 1) # A12 B22
    col += 1
    
    # M6 = (A21-A11)(B11+B12)
    add_term(1, 0, 3, 1) # A21 B11
    add_term(1, 1, 3, 1) # A21 B12
    add_term(0, 0, 0, -1) # A11 B11
    add_term(0, 1, 0, -1) # A11 B12
    col += 1
    
    # M7 = (A12-A22)(B21+B22)
    add_term(1, 4, 1, 1) # A12 B21
    add_term(1, 5, 1, 1) # A12 B22
    add_term(3, 4, 4, -1) # A22 B21
    add_term(3, 5, 4, -1) # A22 B22
    col += 1
    
    # Edge: i=2, j=2, k in 0,1 (1x1x2)
    # U[2*3+2] = U[8], V[2*4+k] = V[8+k], W[k*3+2] = W[2+3k]
    # M8 = A22(B31+B32)
    add_term(8, 8, 2, 1) # A22 B31
    add_term(8, 9, 5, 1) # A22 B32
    col += 1
    
    # M9 = A22(B32-B31)
    add_term(8, 9, 5, 1) # A22 B32
    add_term(8, 8, 2, -1) # A22 B31
    col += 1
    
    # Cross terms (schoolbook)
    # i in 0,1; j=2; k in 0,1
    for i in range(2):
        for k in range(2):
            add_term(i*3+2, 2*4+k, k*3+i, 1)
            col += 1
            
    # i in 0,1; j in 0,1; k=2
    for i in range(2):
        for j in range(2):
            add_term(i*3+j, j*4+2, 2*3+i, 1)
            col += 1
            
    # i in 0,1; j in 0,1; k=3
    for i in range(2):
        for j in range(2):
            add_term(i*3+j, j*4+3, 3*3+i, 1)
            col += 1
            
    # i=2; j in 0,1; k in 0,1
    for j in range(2):
        for k in range(2):
            add_term(2*3+j, j*4+k, k*3+2, 1)
            col += 1
            
    # i=2; j in 0,1; k=2
    for j in range(2):
        add_term(2*3+j, j*4+2, 2*3+2, 1)
        col += 1
        
    # i=2; j in 0,1; k=3
    for j in range(2):
        add_term(2*3+j, j*4+3, 3*3+2, 1)
        col += 1
        
    # i=2; j=2; k=2
    add_term(2*3+2, 2*4+2, 2*3+2, 1)
    col += 1
    
    # i=2; j=2; k=3
    add_term(2*3+2, 2*4+3, 3*3+2, 1)
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
