import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The problem is equivalent to multiplying a 2x2 matrix A by 7 independent 2x2 matrices B_i.
    The minimal rank for 2x2 * 2x2 is 7 (Strassen).
    The total rank is 7 * 7 = 49.
    
    Interface:
    U: shape (n*m, rank) -> (4, 49)
    V: shape (m*p, rank) -> (28, 49)
    W: shape (p*n, rank) -> (28, 49)
    
    Indices:
    A[i, j] corresponds to U[i*m + j, r]
    B[j, k] corresponds to V[j*p + k, r]
    C[i, k] corresponds to W[k*n + i, r] (Trace convention W[k*n+i, r])
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    num_blocks = p // m  # 7 blocks of 2x2
    
    # Strassen's algorithm terms for 2x2 * 2x2
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    rank = num_blocks * 7
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)

    for t in range(num_blocks):
        if time.time() > deadline:
            break
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_k_local = b_idx % 2
                b_k_global = 2 * t + b_k_local
                V[b_j * p + b_k_global, col] = coeff
            
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_k_local = c_idx % 2
                c_k_global = 2 * t + c_k_local
                W[c_k_global * n + c_i, col] = coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
