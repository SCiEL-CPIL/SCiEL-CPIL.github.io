import time
import numpy as np

def construct():
    # Dimensions
    n, m, p = 2, 2, 10
    rank = 35
    
    # Initialize U, V, W as lists of lists of zeros
    # U: (n*m, rank) = (4, 35)
    # V: (m*p, rank) = (20, 35)
    # W: (p*n, rank) = (20, 35)
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = A*B = [[C11, C12], [C21, C22]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(g)
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W mapping per block: w_vec = [C11, C12, C21, C22]
    # W[2*k_block] = C11, W[2*k_block+1] = C12, W[10+2*k_block] = C21, W[10+2*k_block+1] = C22
    
    strassen_decomp = [
        # M1 = (a+d)(e+h)
        (
            [1, 0, 0, 1],   # a+d
            [1, 0, 0, 1],   # e+h
            [1, 0, 0, 1]    # C11:+1, C12:0, C21:0, C22:+1
        ),
        # M2 = (c+d)(g)
        (
            [0, 0, 1, 1],   # c+d
            [0, 0, 1, 0],   # g
            [0, 0, 1, -1]   # C11:0, C12:0, C21:+1, C22:-1
        ),
        # M3 = a(f-h)
        (
            [1, 0, 0, 0],   # a
            [0, 1, 0, -1],  # f-h
            [0, 1, 0, 1]    # C11:0, C12:+1, C21:0, C22:+1
        ),
        # M4 = d(g-f)
        (
            [0, 0, 0, 1],   # d
            [0, -1, 1, 0],  # g-f
            [1, 0, 1, 0]    # C11:+1, C12:0, C21:+1, C22:0
        ),
        # M5 = (a+b)(h)
        (
            [1, 1, 0, 0],   # a+b
            [0, 0, 0, 1],   # h
            [-1, 1, 0, 0]   # C11:-1, C12:+1, C21:0, C22:0
        ),
        # M6 = (c-a)(e+f)
        (
            [-1, 0, 1, 0],  # c-a
            [1, 1, 0, 0],   # e+f
            [0, 0, 0, 1]    # C11:0, C12:0, C21:0, C22:+1
        ),
        # M7 = (b-d)(g+h)
        (
            [0, 1, 0, -1],  # b-d
            [0, 0, 1, 1],   # g+h
            [1, 0, 0, 0]    # C11:+1, C12:0, C21:0, C22:0
        ),
    ]
    
    # Fill the matrices
    col_idx = 0
    for k_block in range(5):
        for u_vec, v_vec, w_vec in strassen_decomp:
            # U: A indices 0..3 -> [a, b, c, d]
            for i in range(4):
                U[i][col_idx] = u_vec[i]
                
            # V: B indices for block k_block
            # B11 -> 2*k_block
            # B12 -> 2*k_block + 1
            # B21 -> 10 + 2*k_block
            # B22 -> 10 + 2*k_block + 1
            V[2 * k_block][col_idx] = v_vec[0]
            V[2 * k_block + 1][col_idx] = v_vec[1]
            V[10 + 2 * k_block][col_idx] = v_vec[2]
            V[10 + 2 * k_block + 1][col_idx] = v_vec[3]
            
            # W: C indices for block k_block
            # C11 -> W[2*k_block]
            # C12 -> W[2*k_block+1]
            # C21 -> W[10+2*k_block]
            # C22 -> W[10+2*k_block+1]
            W[2 * k_block][col_idx] = w_vec[0]
            W[2 * k_block + 1][col_idx] = w_vec[1]
            W[10 + 2 * k_block][col_idx] = w_vec[2]
            W[10 + 2 * k_block + 1][col_idx] = w_vec[3]
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
