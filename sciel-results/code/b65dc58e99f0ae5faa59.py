import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen: 7 outer * 7 inner = 49 rank-1 terms.
    
    U: (16, 49) - U[i*4+j, r] for A[i,j]
    V: (16, 49) - V[j*4+k, r] for B[j,k]
    W: (16, 49) - W[k*4+i, r] for C[i,k] (trace convention)
    """
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 coefficients
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ], dtype=np.int64)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ], dtype=np.int64)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ], dtype=np.int64)

    rank = 49
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            
            # U[row, col] where row = i*4+j for A[i,j]
            # i = I*2+ip, j = J*2+jp
            # coeff = a_coeffs[s][I][J] * a_coeffs[t][ip][jp]
            for I in range(2):
                for J in range(2):
                    as_val = a_coeffs[s, I, J]
                    if as_val == 0:
                        continue
                    for ip in range(2):
                        for jp in range(2):
                            at_val = a_coeffs[t, ip, jp]
                            if at_val == 0:
                                continue
                            coeff = as_val * at_val
                            i = I * 2 + ip
                            j = J * 2 + jp
                            row = i * 4 + j
                            U[row, col] = coeff
            
            # V[row, col] where row = j*4+k for B[j,k]
            # j = J*2+jp, k = K*2+kp
            # coeff = b_coeffs[s][J][K] * b_coeffs[t][jp][kp]
            for J in range(2):
                for K in range(2):
                    bs_val = b_coeffs[s, J, K]
                    if bs_val == 0:
                        continue
                    for jp in range(2):
                        for kp in range(2):
                            bt_val = b_coeffs[t, jp, kp]
                            if bt_val == 0:
                                continue
                            coeff = bs_val * bt_val
                            j = J * 2 + jp
                            k = K * 2 + kp
                            row = j * 4 + k
                            V[row, col] = coeff
            
            # W[row, col] where row = k*4+i for C[i,k] (trace convention)
            # i = I*2+ip, k = K*2+kp
            # coeff = c_coeffs[s][I][K] * c_coeffs[t][ip][kp]
            for I in range(2):
                for K in range(2):
                    cs_val = c_coeffs[s, I, K]
                    if cs_val == 0:
                        continue
                    for ip in range(2):
                        for kp in range(2):
                            ct_val = c_coeffs[t, ip, kp]
                            if ct_val == 0:
                                continue
                            coeff = cs_val * ct_val
                            i = I * 2 + ip
                            k = K * 2 + kp
                            row = k * 4 + i
                            W[row, col] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
