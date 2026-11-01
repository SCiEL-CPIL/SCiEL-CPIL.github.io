import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    # 5 blocks of 2x2
    num_blocks = p // 2 
    # Rank per block is 7
    R = num_blocks * 7
    
    # Initialize matrices as numpy arrays for efficient assignment
    U_np = np.zeros((n*m, R), dtype=int)
    V_np = np.zeros((m*p, R), dtype=int)
    W_np = np.zeros((p*n, R), dtype=int)
    
    # Strassen decomposition vectors
    # Format: (u_vec, v_vec, w_vec)
    # u_vec: coefficients for A (2x2) -> 4 dims
    # v_vec: coefficients for B (2x2) -> 4 dims
    # w_vec: coefficients for C (2x2) -> 4 dims
    strassen_decomp = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))
    ]
    
    col_idx = 0
    for k in range(num_blocks):
        for u, v, w in strassen_decomp:
            # Assign U
            U_np[:, col_idx] = u
            
            # Assign V
            # Block k occupies columns 2*k and 2*k+1 in B (which is 2x10)
            # B is 2x10. Indexing:
            # B[0, 2k], B[0, 2k+1], B[1, 2k], B[1, 2k+1]
            # Flattened V (m*p = 20):
            # Row 0: 0..9
            # Row 1: 10..19
            # So B[0, 2k] is V[2k]
            # B[0, 2k+1] is V[2k+1]
            # B[1, 2k] is V[10+2k]
            # B[1, 2k+1] is V[10+2k+1]
            V_np[2*k, col_idx] = v[0]
            V_np[2*k+1, col_idx] = v[1]
            V_np[10+2*k, col_idx] = v[2]
            V_np[10+2*k+1, col_idx] = v[3]
            
            # Assign W
            # C is 2x10.
            # C[0, 2k], C[0, 2k+1], C[1, 2k], C[1, 2k+1]
            # Flattened W (p*n = 20):
            # Convention W[k*n+i, r]
            # Index = j*n + i
            # C[0, j] -> index j
            # C[1, j] -> index 10 + j
            # So C[0, 2k] is W[2k]
            # C[0, 2k+1] is W[2k+1]
            # C[1, 2k] is W[10+2k]
            # C[1, 2k+1] is W[10+2k+1]
            W_np[2*k, col_idx] = w[0]
            W_np[2*k+1, col_idx] = w[1]
            W_np[10+2*k, col_idx] = w[2]
            W_np[10+2*k+1, col_idx] = w[3]
            
            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
