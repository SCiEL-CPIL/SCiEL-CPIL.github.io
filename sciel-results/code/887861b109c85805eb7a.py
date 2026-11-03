import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Block decomposition: 3x3 and 2x2
    # Ranks:
    # A11*B11 (3x3): 27
    # A12*B21 (3x2, 2x3): 18
    # A11*B12 (3x3, 3x2): 18
    # A12*B22 (3x2, 2x2): 12
    # A21*B11 (2x3, 3x3): 18
    # A22*B21 (2x2, 2x3): 12
    # A21*B12 (2x3, 3x2): 12
    # A22*B22 (2x2): 8
    # Total: 107
    
    rank = 107
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # 1. A11*B11 -> C11
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_term(i, j, k)
                
    # 2. A12*B21 -> C11
    # A12: i in 0..2, j in 3..4
    # B21: j in 3..4, k in 0..2
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                add_term(i, j, k)
                
    # 3. A11*B12 -> C12
    # A11: i in 0..2, j in 0..2
    # B12: j in 0..2, k in 3..4
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                add_term(i, j, k)
                
    # 4. A12*B22 -> C12
    # A12: i in 0..2, j in 3..4
    # B22: j in 3..4, k in 3..4
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 5):
                add_term(i, j, k)
                
    # 5. A21*B11 -> C21
    # A21: i in 3..4, j in 0..2
    # B11: j in 0..2, k in 0..2
    for i in range(3, 5):
        for j in range(3):
            for k in range(3):
                add_term(i, j, k)
                
    # 6. A22*B21 -> C21
    # A22: i in 3..4, j in 3..4
    # B21: j in 3..4, k in 0..2
    for i in range(3, 5):
        for j in range(3, 5):
            for k in range(3):
                add_term(i, j, k)
                
    # 7. A21*B12 -> C22
    # A21: i in 3..4, j in 0..2
    # B12: j in 0..2, k in 3..4
    for i in range(3, 5):
        for j in range(3):
            for k in range(3, 5):
                add_term(i, j, k)
                
    # 8. A22*B22 -> C22
    # A22: i in 3..4, j in 3..4
    # B22: j in 3..4, k in 3..4
    for i in range(3, 5):
        for j in range(3, 5):
            for k in range(3, 5):
                add_term(i, j, k)
                
    return {"U": U, "V": V, "W": W}
