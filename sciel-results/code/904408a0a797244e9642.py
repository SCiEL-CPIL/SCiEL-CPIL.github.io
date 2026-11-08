import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 120.0

def _is_timeout():
    return (time.time() - _START_TIME) > _TIME_LIMIT

DIMENSIONS = [4, 4, 8]

def construct():
    if _is_timeout():
        # Fallback to naive decomposition rank 128
        U = [[0]*128 for _ in range(16)]
        V = [[0]*128 for _ in range(32)]
        W = [[0]*128 for _ in range(32)]
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j][r] = 1
                    V[j*8+k][r] = 1
                    W[k*4+i][r] = 1
        return {"U": U, "V": V, "W": W}

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Strassen base case for 2x2 (rank 7)
    # Indices: 00->0, 01->1, 10->2, 11->3
    base_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    base_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    base_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Build 4x4 (rank 49) using Kronecker product
    # u_4x4[r] will have length 16
    # v_4x4[r] will have length 16
    # w_4x4[r] will have length 16
    
    u_4x4 = []
    v_4x4 = []
    w_4x4 = []
    
    for s in range(7):
        for t in range(7):
            u_new = np.kron(np.array(base_u[s]), np.array(base_u[t]))
            v_new = np.kron(np.array(base_v[s]), np.array(base_v[t]))
            w_new = np.kron(np.array(base_w[s]), np.array(base_w[t]))
            u_4x4.append(u_new)
            v_4x4.append(v_new)
            w_4x4.append(w_new)
            
    rank_4x4 = 49
    total_rank = 98
    
    # Initialize matrices
    # U: (n*m, rank) = (16, 98)
    # V: (m*p, rank) = (32, 98)
    # W: (p*n, rank) = (32, 98)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Verify Strassen 4x4 correctness locally first? 
    # Let's just construct.
    
    # Fill first 49 terms (for k in 0..3)
    for r in range(rank_4x4):
        u_vec = u_4x4[r]
        v_vec = v_4x4[r]
        w_vec = w_4x4[r]
        
        # U: indices i*m + j. u_vec is length 16 (i*4+j).
        for idx in range(16):
            U[idx][r] = int(u_vec[idx])
            
        # V: indices j*p + k. v_vec is length 16 (j*4+k).
        # For k in 0..3, V[j*8+k] = v_vec[j*4+k]
        for j in range(4):
            for k in range(4):
                V[j * 8 + k][r] = int(v_vec[j * 4 + k])
                
        # W: indices k*n + i. w_vec is length 16 (k*4+i).
        # For k in 0..3, W[k*4+i] = w_vec[k*4+i]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][r] = int(w_vec[k * 4 + i])
                
    # Fill next 49 terms (for k in 4..7)
    for r in range(rank_4x4):
        u_vec = u_4x4[r]
        v_vec = v_4x4[r]
        w_vec = w_4x4[r]
        
        col = r + rank_4x4
        
        # U: same as before
        for idx in range(16):
            U[idx][col] = int(u_vec[idx])
            
        # V: Map v_vec[j*4 + k] to V[j*8 + (k+4)]
        for j in range(4):
            for k in range(4):
                V[j * 8 + (k + 4)][col] = int(v_vec[j * 4 + k])
                
        # W: Map w_vec[k*4 + i] to W[(k+4)*4 + i]
        for k in range(4):
            for i in range(4):
                W[(k + 4) * 4 + i][col] = int(w_vec[k * 4 + i])
                
    return {"U": U, "V": V, "W": W}
