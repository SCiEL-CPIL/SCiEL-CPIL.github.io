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
    # C = [[i, j], [k, l]]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6
    
    # We represent vectors for A, B, C as [A11, A12, A21, A22]
    # A_vec: [a, b, c, d]
    # B_vec: [e, f, g, h]
    # C_vec: [i, j, k, l]
    
    # Each entry in the tensor decomposition is a rank-1 term u * v * w
    # u is from A space, v from B space, w from C space.
    
    strassen_decomp = [
        # M1
        (
            [1, 0, 0, 1],  # a+d
            [1, 0, 0, 1],  # e+h
            [1, 0, 0, 1]   # contributes to i, j, k, l? 
                           # i: +1, j: 0, k: 0, l: +1? 
                           # Wait, C_vec order in seed was [C11, C21, C12, C22] which is [i, k, j, l]
                           # Let's check seed W mapping:
                           # W[4k] = c_vec[0] -> C11 (i)
                           # W[4k+1] = c_vec[1] -> C21 (k)
                           # W[4k+2] = c_vec[2] -> C12 (j)
                           # W[4k+3] = c_vec[3] -> C22 (l)
                           # So C_vec is [i, k, j, l]
        ),
        (
            [0, 0, 1, 1],  # c+d
            [1, 0, 0, 0],  # e
            [0, 1, 0, -1]  # k: +1, l: -1
        ),
        (
            [1, 0, 0, 0],  # a
            [0, 1, 0, -1], # f-h
            [0, 0, 1, 1]   # j: +1, l: +1
        ),
        (
            [0, 0, 0, 1],  # d
            [-1, 0, 1, 0], # g-f
            [1, 1, 0, 0]   # i: +1, k: +1
        ),
        (
            [1, 1, 0, 0],  # a+b
            [0, 0, 0, 1],  # h
            [-1, 0, 1, 0]  # i: -1, j: +1
        ),
        (
            [-1, 0, 1, 0], # c-a
            [1, 1, 0, 0],  # e+f
            [0, 0, 0, 1]   # l: +1
        ),
        (
            [0, 1, 0, -1], # b-d
            [0, 0, 1, 1],  # g+h
            [1, 0, 0, 0]   # i: +1
        ),
    ]
    
    # Fill the matrices
    col_idx = 0
    for k_block in range(5):
        for u_vec, v_vec, w_vec in strassen_decomp:
            # U: A indices 0..3
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
            # C11 -> 4*k_block
            # C21 -> 4*k_block + 1
            # C12 -> 4*k_block + 2
            # C22 -> 4*k_block + 3
            W[4 * k_block][col_idx] = w_vec[0]
            W[4 * k_block + 1][col_idx] = w_vec[1]
            W[4 * k_block + 2][col_idx] = w_vec[2]
            W[4 * k_block + 3][col_idx] = w_vec[3]
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
