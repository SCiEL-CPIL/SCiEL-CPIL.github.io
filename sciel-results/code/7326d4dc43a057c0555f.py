import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients
    A_forms = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    
    B_forms_local = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    
    W_forms_local = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    col = 0
    for k in range(7):
        for s in range(7):
            U[:, col] = A_forms[s]
            
            local_indices_v = [2*k, 2*k+1, 14+2*k, 14+2*k+1]
            for i in range(4):
                if B_forms_local[s][i] != 0:
                    V[local_indices_v[i], col] = B_forms_local[s][i]
            
            local_indices_w = [4*k, 4*k+1, 4*k+2, 4*k+3]
            for i in range(4):
                if W_forms_local[s][i] != 0:
                    W[local_indices_w[i], col] = W_forms_local[s][i]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
