"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # We use a hybrid decomposition:
    # - For 12 of the 13 columns (organized as 6 blocks of 2 columns), we use a rank-7
    #   exact decomposition for 2x2 by 2x2 matrix multiplication.
    # - For the remaining 1 column, we use the schoolbook rank-4 decomposition.
    # Total rank = 6 * 7 + 4 = 46.
    
    rank = 46
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    # Strassen-based decomposition for 2x2 by 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[C00, C01], [C10, C11]]
    #
    # M1 = (a+d)(e+h)  -> C00 += M1, C11 += M1
    # M2 = (c+d)e      -> C10 += M2, C11 -= M2
    # M3 = a(f-h)      -> C01 += M3, C11 += M3
    # M4 = d(g-e)      -> C00 += M4, C10 += M4
    # M5 = (a+b)h      -> C00 -= M5, C01 += M5
    # M6 = (c-d)(f+g)  -> C11 += M6
    # M7 = (b-d)(e+g)  -> C00 += M7
    
    # Strassen coefficients: (U_coeffs, V_coeffs, W_coeffs)
    # U is for A (4 elements: a, b, c, d)
    # V is for B (4 elements: e, f, g, h)
    # W is for C (4 elements: C00, C01, C10, C11)
    
    strassen_terms = [
        # M1: (a+d)(e+h)
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2: (c+d)e
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        # M3: a(f-h)
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
        # M4: d(g-e)
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])),
        # M5: (a+b)h
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        # M6: (c-d)(f+g)
        (np.array([0, 0, 1, -1]), np.array([0, 1, 1, 0]), np.array([0, 0, 0, 1])),
        # M7: (b-d)(e+g)
        (np.array([0, 1, 0, -1]), np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0])),
    ]
    
    col = 0
    # Process 6 blocks of 2 columns each
    for block in range(6):
        k0 = 2 * block  # first column index in this block
        k1 = k0 + 1     # second column index
        
        for (u_coeff, v_coeff, w_coeff) in strassen_terms:
            # U: fill in the 4x1 block for column col
            for i in range(n):
                for j in range(m):
                    a_idx = i * m + j
                    U[a_idx, col] = u_coeff[i * m + j]
            
            # V: fill in the 4x1 block for columns k0 and k1
            # V has shape (m*p, rank), so index is j*p + k
            for j in range(m):
                for k_idx, k_val in enumerate([k0, k1]):
                    b_idx = j * p + k_val
                    V[b_idx, col] = v_coeff[j * 2 + k_idx]
            
            # W: fill in the 4x1 block for rows k0 and k1
            # W has shape (p*n, rank), so index is k*n + i
            for k_idx, k_val in enumerate([k0, k1]):
                for i in range(n):
                    c_idx = k_val * n + i
                    W[c_idx, col] = w_coeff[i * 2 + k_idx]
            
            col += 1
    
    # Schoolbook for the last column (k=12)
    k_last = 12
    for i in range(n):
        for j in range(m):
            a_idx = i * m + j
            b_idx = j * p + k_last
            c_idx = k_last * n + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

