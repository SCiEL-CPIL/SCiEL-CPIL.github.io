import numpy as np

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
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    num_blocks = p // m  # 7 blocks of 2x2
    
    # Strassen's algorithm terms for 2x2 * 2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Each term is (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are dicts {index: value}
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01 - B11) -> C01 += M3, C11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10 - B00) -> C00 += M4, C10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    rank = num_blocks * 7
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            # U: A[i, j] -> index i*m + j
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            # V: B[j, k] -> index j*p + k
            # Block t covers columns 2*t and 2*t+1
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2 # Row of B block (0 or 1)
                b_k_local = b_idx % 2 # Col of B block (0 or 1)
                b_k_global = 2 * t + b_k_local # Global col of B
                V[b_j * p + b_k_global, col] = coeff
            
            # W: C[i, k] -> index k*n + i
            # Block t covers columns 2*t and 2*t+1
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2 # Row of C block
                c_k_local = c_idx % 2 # Col of C block
                c_k_global = 2 * t + c_k_local # Global col of C
                W[c_k_global * n + c_i, col] = coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
