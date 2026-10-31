import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define the rank-7 Strassen decomposition for 2x2 matrices
    # U2, V2, W2 are 4x7 matrices
    # Indices: 00, 01, 10, 11
    
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0], # a
        [0, 0, 0, 0, 0, 1, 1], # b
        [0, 1, 0, 0, 1, 0, 1], # c
        [1, 1, 0, 1, 0, 1, 0]  # d
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, -1, 0, 0, 0, 1], # e
        [1, 0, 1, 0, 0, 0, 1],  # f
        [0, 0, 0, -1, 0, 1, 0], # g
        [0, 0, 0, 1, 1, 0, 0]   # h
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ], dtype=float)
    
    # Construct rank-49 decomposition for 4x4 matrices
    # U4 is 16x49
    # Row index: (i1*2 + i0)*4 + (j1*2 + j0)
    # Col index: R*7 + S
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    row_U = (i1 * 2 + i0) * 4 + (j1 * 2 + j0)
                    
                    for R in range(7):
                        for S in range(7):
                            col = R * 7 + S
                            
                            # U4
                            u_val = U2[i1 * 2 + i0, R] * U2[j1 * 2 + j0, S]
                            U4[row_U, col] = u_val
                            
                            # V4
                            # V index: (j1*2+j0)*4 + (k1*2+k0)
                            # But V2 is indexed by j, k.
                            # V2[j, r] corresponds to B_{j,k}?
                            # In 2x2, V2 row is j*2+k.
                            # Here row_V = (j1*2 + j0)*4 + (k1*2 + k0)
                            # V4[row_V, col] = V2[j1*2+j0, R] * V2[k1*2+k0, S]
                            # We need to fill V4 for all k1, k0
                            for k1 in range(2):
                                for k0 in range(2):
                                    row_V = (j1 * 2 + j0) * 4 + (k1 * 2 + k0)
                                    v_val = V2[j1 * 2 + j0, R] * V2[k1 * 2 + k0, S]
                                    V4[row_V, col] = v_val
                                    
                                    # W4
                                    # W index: (k1*2+k0)*4 + (i1*2+i0)
                                    row_W = (k1 * 2 + k0) * 4 + (i1 * 2 + i0)
                                    w_val = W2[k1 * 2 + k0, R] * W2[i1 * 2 + i0, S]
                                    W4[row_W, col] = w_val

    # Convert to lists of lists as required
    U = U4.tolist()
    V = V4.tolist()
    W = W4.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
