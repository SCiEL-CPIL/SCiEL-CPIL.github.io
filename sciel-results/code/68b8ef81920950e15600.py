import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS

    # Strassen coefficients for 2x2 matrices (0-indexed):
    # s=0..6
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: A00 + A11
        [[0, 0], [1, 1]],   # s=1: A10 + A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00 + A01
        [[-1, 0], [1, 0]],  # s=5: A10 - A00
        [[0, 1], [0, -1]],  # s=6: A01 - A11
    ], dtype=int)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: B00 + B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01 - B11
        [[-1, 0], [1, 0]],  # s=3: B10 - B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00 + B01
        [[0, 0], [1, 1]],   # s=6: B10 + B11
    ], dtype=int)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0: C00 + C11
        [[0, 0], [1, -1]],  # s=1: C10 - C11
        [[0, 1], [0, 1]],   # s=2: C01 + C11
        [[1, 0], [1, 0]],   # s=3: C00 + C10
        [[-1, 1], [0, 0]],  # s=4: -C00 + C01
        [[0, 0], [0, 1]],   # s=5: C11
        [[1, 0], [0, 0]],   # s=6: C00
    ], dtype=int)

    # Build 4x4x4 rank-49 decomposition via Kronecker products of 2x2 Strassen
    U_4d = np.zeros((7, 7, 4, 4), dtype=int)
    V_4d = np.zeros((7, 7, 4, 4), dtype=int)
    W_4d = np.zeros((7, 7, 4, 4), dtype=int)
    
    for s in range(7):
        for t in range(7):
            U_4d[s, t] = np.kron(a_coeffs[s], a_coeffs[t])
            V_4d[s, t] = np.kron(b_coeffs[s], b_coeffs[t])
            # W: row index k*4+i for C[i,k] per trace convention W[k*n+i, r]
            # kron(c_s, c_t)[i, k] gives coeff for C[i,k]
            # We need W[k*4+i, s*7+t] = coeff of C[i,k]
            # So W_4d[s,t] should have [k, i] = kron(c_s, c_t)[i, k]
            # i.e., transpose
            W_4d[s, t] = np.kron(c_coeffs[s], c_coeffs[t]).T
            
            if time.time() > deadline - 5.0:
                # Return the rank-49 result immediately
                U = U_4d.transpose(2, 3, 0, 1).reshape(16, 49)
                V = V_4d.transpose(2, 3, 0, 1).reshape(16, 49)
                W = W_4d.transpose(2, 3, 0, 1).reshape(16, 49)
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Flatten to (16, 49)
    U = U_4d.transpose(2, 3, 0, 1).reshape(16, 49)
    V = V_4d.transpose(2, 3, 0, 1).reshape(16, 49)
    W = W_4d.transpose(2, 3, 0, 1).reshape(16, 49)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
