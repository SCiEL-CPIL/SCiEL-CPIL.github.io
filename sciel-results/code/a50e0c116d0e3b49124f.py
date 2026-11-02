import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The tensor is the direct sum of 7 copies of the 2x2 * 2x2 multiplication tensor.
    The rank of 2x2 * 2x2 is 7 (Strassen).
    The rank of the direct sum is the sum of the ranks: 7 * 7 = 49.
    This is the minimal rank for this tensor.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin out of 120s
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    num_blocks = p // m # 14 // 2 = 7
    
    # Strassen terms for 2x2 * 2x2
    # A indices: 00, 01, 10, 11 (0, 1, 2, 3)
    # B indices: 00, 01, 10, 11 (0, 1, 2, 3)
    # C indices: 00, 01, 10, 11 (0, 1, 2, 3)
    
    # Each term is (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are dicts {index: value}
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),        # M7
    ]

    rank = num_blocks * 7
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    for t in range(num_blocks):
        if time.time() > deadline:
            # Return partial result if time runs out, though unlikely
            U_list = [[int(round(x)) for x in row] for row in U.tolist()]
            V_list = [[int(round(x)) for x in row] for row in V.tolist()]
            W_list = [[int(round(x)) for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}

        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            for b_idx, coeff in b_coeffs.items():
                b_i = b_idx // 2 # Row of B block (0 or 1)
                b_j_local = b_idx % 2 # Col of B block (0 or 1)
                b_k = 2 * t + b_j_local # Global col of B
                V[b_i * p + b_k, col] = coeff
            
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2 # Row of C block
                c_j_local = c_idx % 2 # Col of C block
                c_k = 2 * t + c_j_local # Global col of C
                W[c_k * n + c_i, col] = coeff

    # Convert to lists of integers
    U_list = [[int(round(x)) for x in row] for row in U.tolist()]
    V_list = [[int(round(x)) for x in row] for row in V.tolist()]
    W_list = [[int(round(x)) for x in row] for row in W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
