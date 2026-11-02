import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    This is the known minimal rank for this tensor.
    Uses 7 Strassen terms for each of the 7 2x2 blocks of the B matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 2, 2, 14
    nm = n * m
    mp = m * p
    pn = p * n
    num_blocks = p // 2  # 7 blocks of 2 columns
    rank = num_blocks * 7  # 49

    U = np.zeros((nm, rank), dtype=np.int8)
    V = np.zeros((mp, rank), dtype=np.int8)
    W = np.zeros((pn, rank), dtype=np.int8)

    # Strassen terms for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A B
    #
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b11 - b00)
    # m4 = a11 * (b00 - b10)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Coefficients: (index_in_matrix, coeff)
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11

    strassen_terms = [
        # (U coeffs, V coeffs, W coeffs)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # m1
        ([(2, 1), (3, 1)], [(0, 1)],         [(2, 1), (3, -1)]),   # m2
        ([(0, 1)],         [(1, 1), (3, -1)],[(1, 1), (3, 1)]),    # m3
        ([(3, 1)],         [(2, 1), (0, -1)],[(0, 1), (2, 1)]),    # m4
        ([(0, 1), (1, 1)], [(3, 1)],         [(0, -1), (1, 1)]),   # m5
        ([(2, 1), (0, -1)],[(0, 1), (1, 1)], [(3, 1)]),            # m6
        ([(1, 1), (3, -1)],[(2, 1), (3, 1)], [(0, 1)]),            # m7
    ]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            for a_idx, coeff in u_coeffs:
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            for b_idx, coeff in v_coeffs:
                b_j = b_idx // 2
                b_col = b_idx % 2
                # B is 2x14. Block t covers columns 2t, 2t+1.
                # b_idx 0,1,2,3 correspond to B[0, 2t], B[0, 2t+1], B[1, 2t], B[1, 2t+1]
                # b_j is row (0 or 1)
                # b_col is col within block (0 or 1)
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff
            
            for c_idx, coeff in w_coeffs:
                c_i = c_idx // 2
                c_col = c_idx % 2
                # C is 2x14. Block t covers columns 2t, 2t+1.
                # c_idx 0,1,2,3 correspond to C[0, 2t], C[0, 2t+1], C[1, 2t], C[1, 2t+1]
                # c_i is row (0 or 1)
                # c_col is col within block (0 or 1)
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
