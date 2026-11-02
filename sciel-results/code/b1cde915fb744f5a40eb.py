import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Rank is 3*3*3*7 = 189
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 matrices
    # Flattened X: [x00, x01, x10, x11]
    # Flattened Y: [y00, y01, y10, y11]
    # Flattened Z: [z00, z01, z10, z11]
    
    u_coeffs = [
        [1, 0, 0, 1],  # S1
        [0, 0, 1, 1],  # S2
        [1, 0, 0, 0],  # S3
        [0, 0, 0, 1],  # S4
        [1, 1, 0, 0],  # S5
        [-1, 0, 1, 0], # S6
        [0, 1, 0, -1]  # S7
    ]
    
    v_coeffs = [
        [1, 0, 0, 1],  # S1
        [1, 0, 0, 0],  # S2
        [0, 1, 0, -1], # S3
        [0, 0, 1, -1], # S4
        [0, 0, 0, 1],  # S5
        [1, 1, 0, 0],  # S6
        [0, 0, 1, 1]   # S7
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],   # S1
        [0, 0, 1, -1],  # S2
        [0, 1, 0, 1],   # S3
        [1, 0, 1, 0],   # S4
        [-1, 1, 0, 0],  # S5
        [0, 0, 0, 1],   # S6
        [1, 0, 0, 0]    # S7
    ]
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for m in range(7):
                    if time.time() > deadline:
                        # Fallback to schoolbook if timeout (unlikely)
                        return {"U": np.zeros((n*m, n*m*p)).tolist(), 
                                "V": np.zeros((m*p, n*m*p)).tolist(), 
                                "W": np.zeros((p*n, n*m*p)).tolist()}
                    
                    # U: A[I, J] block
                    for u in range(2):
                        for v in range(2):
                            a_idx = (2*I + u) * m + (2*J + v)
                            U[a_idx, col] = u_coeffs[m][u*2 + v]
                            
                    # V: B[J, K] block
                    for u in range(2):
                        for v in range(2):
                            b_idx = (2*J + u) * p + (2*K + v)
                            V[b_idx, col] = v_coeffs[m][u*2 + v]
                            
                    # W: C[I, K] block
                    for u in range(2):
                        for v in range(2):
                            c_row = 2*I + u
                            c_col = 2*K + v
                            w_idx = c_col * n + c_row
                            W[w_idx, col] = w_coeffs[m][u*2 + v]
                            
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
