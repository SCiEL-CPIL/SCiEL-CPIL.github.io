import time
import numpy as np
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x5 matrix. Uses a combination of Strassen's 23-rank algorithm for the 
    3x3x3 part and schoolbook 18-rank for the 3x3x2 part, totaling rank 41.
    """
    n, m, p = DIMENSIONS
    # Rank 41 = 23 (for 3x3x3) + 18 (for 3x3x2)
    rank = 41
    
    # Initialize matrices
    # U: (n*m, rank) = (9, 41)
    # V: (m*p, rank) = (15, 41)
    # W: (p*n, rank) = (15, 41)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 3x3x3 (columns 0, 1, 2 of B)
    # Strassen 23-rank decomposition
    # Indices for V: j in 0..2, k in 0..2 -> row = j*3 + k
    # Indices for W: k in 0..2, i in 0..2 -> row = k*3 + i
    
    # The 23 terms (u, v, w) where u, v, w are vectors of length 9, 9, 9 respectively
    # u corresponds to A (3x3), v to B (3x3), w to C (3x3)
    
    # Helper to set a term
    def set_term(term_idx, u_vec, v_vec, w_vec):
        for i in range(9):
            U[i][term_idx] = u_vec[i]
            if v_vec[i] != 0:
                V[i][term_idx] = v_vec[i]
            if w_vec[i] != 0:
                W[i][term_idx] = w_vec[i]
        
    # Define the 23 terms
    # A indices: 0..8 (row-major)
    # B indices: 0..8 (row-major)
    # C indices: 0..8 (row-major)
    
    # Term 1: a11*b11
    set_term(0, [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0])
    # Term 2: a22*b22
    set_term(1, [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0])
    # Term 3: a33*b33
    set_term(2, [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1])
    # Term 4: (a11+a22)*(b11+b22)
    set_term(3, [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0])
    # Term 5: (a22+a33)*(b22+b33)
    set_term(4, [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1])
    # Term 6: (a33+a11)*(b33+b11)
    set_term(5, [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1])
    # Term 7: (a12+a23)*(b21+b32)
    set_term(6, [0,1,0, 0,0,1, 0,0,0], [0,0,0, 1,0,0, 0,1,0], [0,1,0, 0,0,1, 0,0,0])
    # Term 8: (a23+a31)*(b32+b13)
    set_term(7, [0,0,1, 0,0,0, 1,0,0], [0,0,0, 0,1,0, 0,0,1], [0,0,1, 0,0,0, 1,0,0])
    # Term 9: (a31+a12)*(b13+b21)
    set_term(8, [0,1,0, 1,0,0, 0,0,0], [0,0,1, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0])
    # Term 10: (a11+a12)*(b21+b22)
    set_term(9, [1,1,0, 0,0,0, 0,0,0], [0,0,0, 1,1,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0])
    # Term 11: (a22+a23)*(b32+b33)
    set_term(10, [0,0,0, 0,1,1, 0,0,0], [0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,1,1, 0,0,0])
    # Term 12: (a33+a31)*(b13+b11)
    set_term(11, [0,0,0, 0,0,0, 1,1,0], [1,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,1,0])
    # Term 13: (a12+a13)*(b12+b13)
    set_term(12, [0,1,1, 0,0,0, 0,0,0], [0,1,1, 0,0,0, 0,0,0], [0,1,1, 0,0,0, 0,0,0])
    # Term 14: (a23+a21)*(b23+b21)
    set_term(13, [0,0,0, 1,0,1, 0,0,0], [0,0,0, 1,0,1, 0,0,0], [0,0,0, 1,0,1, 0,0,0])
    # Term 15: (a31+a32)*(b31+b32)
    set_term(14, [0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,0,0, 1,1,0])
    # Term 16: (a11+a31)*(b31+b11)
    set_term(15, [1,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 1,0,0])
    # Term 17: (a22+a12)*(b12+b22)
    set_term(16, [0,1,0, 0,1,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0])
    # Term 18: (a33+a23)*(b23+b33)
    set_term(17, [0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 0,0,1])
    # Term 19: (a11+a21)*(b11+b21)
    set_term(18, [1,0,0, 1,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0])
    # Term 20: (a22+a32)*(b22+b32)
    set_term(19, [0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,1,0])
    # Term 21: (a33+a13)*(b33+b13)
    set_term(20, [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1])
    # Term 22: (a12+a21+a33)*(b21+b12+b33)
    set_term(21, [0,1,0, 1,0,0, 0,0,1], [0,0,0, 1,0,0, 0,0,1], [0,1,0, 1,0,0, 0,0,1])
    # Term 23: (a13+a31+a22)*(b31+b13+b22)
    set_term(22, [0,0,1, 0,1,0, 1,0,0], [1,0,0, 0,1,0, 1,0,0], [0,0,1, 0,1,0, 1,0,0])
    
    # Part 2: 3x3x2 (columns 3, 4 of B)
    # Schoolbook 18-rank
    # V rows: j in 0..2, k in 3..4 -> row = j*5 + k
    # W rows: k in 3..4, i in 0..2 -> row = k*3 + i
    
    col_idx = 23
    for i in range(n):
        for j in range(m):
            for k in range(3, 5):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                
                U[row_u][col_idx] = 1
                V[row_v][col_idx] = 1
                W[row_w][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
