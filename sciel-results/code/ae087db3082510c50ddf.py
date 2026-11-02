import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Strassen decomposition for 2x2x2
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W_S = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    num_blocks = p // 2
    
    for l in range(num_blocks):
        col_start = l * 7
        col_end = col_start + 7
        
        # U is shared across blocks
        U[:, col_start:col_end] = U_S
        
        # V mapping:
        # Local 0 -> Global 2l
        # Local 1 -> Global 2l+1
        # Local 2 -> Global 8+2l
        # Local 3 -> Global 8+2l+1
        V[2*l, col_start:col_end] = V_S[0]
        V[2*l+1, col_start:col_end] = V_S[1]
        V[8+2*l, col_start:col_end] = V_S[2]
        V[8+2*l+1, col_start:col_end] = V_S[3]
        
        # W mapping:
        # Local 0 (C_00) -> Global 4l
        # Local 1 (C_01) -> Global 4l+2
        # Local 2 (C_10) -> Global 4l+1
        # Local 3 (C_11) -> Global 4l+3
        W[4*l, col_start:col_end] = W_S[0]
        W[4*l+1, col_start:col_end] = W_S[2]
        W[4*l+2, col_start:col_end] = W_S[1]
        W[4*l+3, col_start:col_end] = W_S[3]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
