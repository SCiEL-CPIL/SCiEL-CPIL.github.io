import time
import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Returns a rank-49 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix using Strassen's algorithm
    applied to 7 blocks of 2 columns each.
    
    U: (4, 49) - linear forms on A
    V: (28, 49) - linear forms on B  
    W: (28, 49) - linear forms on C (W[k*n+i, r] convention)
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    U = np.zeros((n * m, rank))   # (4, 49)
    V = np.zeros((m * p, rank))   # (28, 49)
    W = np.zeros((p * n, rank))   # (28, 49)
    
    # Strassen terms for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B_block = [[e, f], [g, h]]
    # 7 products and their contributions to C_block = [[c11, c12], [c21, c22]]:
    #
    # M1 = (a+d)(e+h)  -> c11+=1, c22+=1
    # M2 = (c+d)e      -> c21+=1, c22-=-1
    # M3 = a(f-h)      -> c12+=1, c22+=1
    # M4 = d(g-e)      -> c11+=1, c21+=1
    # M5 = (a+b)h      -> c11-=-1, c12+=1
    # M6 = (c-a)(e+f)  -> c22+=1
    # M7 = (b-d)(g+h)  -> c11+=1
    #
    # Verification:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Strassen term definitions: (a, b, c, d, e, f, g, h, c11, c12, c21, c22)
    strassen_terms = [
        ( 1, 0, 0, 1,  1, 0, 0, 1,  1,  0, 0,  1),  # M1 = (a+d)(e+h)
        ( 0, 0, 1, 1,  1, 0, 0, 0,  0,  0, 1, -1),  # M2 = (c+d)e
        ( 1, 0, 0, 0,  0, 1, 0,-1,  0,  1, 0,  1),  # M3 = a(f-h)
        ( 0, 0, 0, 1, -1, 0, 1, 0,  1,  0, 1,  0),  # M4 = d(g-e)
        ( 1, 1, 0, 0,  0, 0, 0, 1, -1,  1, 0,  0),  # M5 = (a+b)h
        (-1, 0, 1, 0,  1, 1, 0, 0,  0,  0, 0,  1),  # M6 = (c-a)(e+f)
        ( 0, 1, 0,-1,  0, 0, 1, 1,  1,  0, 0,  0),  # M7 = (b-d)(g+h)
    ]
    
    for block in range(num_blocks):
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c11_c, c12_c, c21_c, c22_c = term
            
            # U: A[i,j] at index i*m + j
            U[0, col] = a_c   # A[0,0]
            U[1, col] = b_c   # A[0,1]
            U[2, col] = c_c   # A[1,0]
            U[3, col] = d_c   # A[1,1]
            
            # V: B[j,k] at index j*p + k
            V[0 * p + 2 * block, col] = e_c       # B[0, 2b]
            V[0 * p + 2 * block + 1, col] = f_c   # B[0, 2b+1]
            V[1 * p + 2 * block, col] = g_c       # B[1, 2b]
            V[1 * p + 2 * block + 1, col] = h_c   # B[1, 2b+1]
            
            # W: C[i,k] at index k*n + i (trace convention)
            W[(2 * block) * n + 0, col] = c11_c     # C[0, 2b]
            W[(2 * block) * n + 1, col] = c21_c     # C[1, 2b]
            W[(2 * block + 1) * n + 0, col] = c12_c # C[0, 2b+1]
            W[(2 * block + 1) * n + 1, col] = c22_c # C[1, 2b+1]
    
    return {"U": U, "V": V, "W": W}
