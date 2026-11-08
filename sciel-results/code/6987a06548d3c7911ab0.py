import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Rank 36 solution
    R = 36
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    rank_offset = 0
    
    # Strassen terms for 2x2x2
    # U: A indices (i, j) -> i*2+j
    # V: B indices (j, k) -> j*2+k
    # W: C indices (k, i) -> k*2+i
    strassen_terms = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, 1, -1, 0]), np.array([0, 0, 1, 1])),
        # M4
        (np.array([0, 0, 0, 1]), np.array([0, 0, 1, -1]), np.array([1, 1, 0, 0])),
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
        # M6
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        # M7
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))
    ]
    
    # Simple terms for 2x2x1 (Rank 4)
    # U: A indices (i, j) -> i*2+j
    # V: B indices (j, 0) -> j
    # W: C indices (0, i) -> i
    simple_terms = [
        # A00 * B00 -> C00
        (np.array([1, 0, 0, 0]), np.array([1, 0]), np.array([1, 0])),
        # A01 * B10 -> C00
        (np.array([0, 1, 0, 0]), np.array([0, 1]), np.array([1, 0])),
        # A10 * B00 -> C10
        (np.array([0, 0, 1, 0]), np.array([1, 0]), np.array([0, 1])),
        # A11 * B10 -> C10
        (np.array([0, 0, 0, 1]), np.array([0, 1]), np.array([0, 1]))
    ]
    
    def add_strassen_block(j_off, k_off):
        nonlocal rank_offset
        for u_vec, v_vec, w_vec in strassen_terms:
            col = rank_offset
            
            # U
            for idx, val in enumerate(u_vec):
                if val != 0:
                    i = idx // 2
                    j = idx % 2
                    u_idx = i * m + (j_off + j)
                    U[u_idx, col] = val
            
            # V
            for idx, val in enumerate(v_vec):
                if val != 0:
                    j = idx // 2
                    k = idx % 2
                    v_idx = (j_off + j) * p + (k_off + k)
                    V[v_idx, col] = val
                    
            # W
            for idx, val in enumerate(w_vec):
                if val != 0:
                    k = idx // 2
                    i = idx % 2
                    w_idx = (k_off + k) * n + i
                    W[w_idx, col] = val
                    
            rank_offset += 1

    def add_simple_block(j_off, k_off):
        nonlocal rank_offset
        for u_vec, v_vec, w_vec in simple_terms:
            col = rank_offset
            
            # U
            for idx, val in enumerate(u_vec):
                if val != 0:
                    i = idx // 2
                    j = idx % 2
                    u_idx = i * m + (j_off + j)
                    U[u_idx, col] = val
            
            # V
            for idx, val in enumerate(v_vec):
                if val != 0:
                    j = idx
                    k = 0
                    v_idx = (j_off + j) * p + (k_off + k)
                    V[v_idx, col] = val
                    
            # W
            for idx, val in enumerate(w_vec):
                if val != 0:
                    k = 0
                    i = idx
                    w_idx = (k_off + k) * n + i
                    W[w_idx, col] = val
                    
            rank_offset += 1

    # Blocks
    # j_off in {0, 2}, k_off in {0, 2, 4}
    # k_off=0: 2x2x2 (Strassen)
    # k_off=2: 2x2x2 (Strassen)
    # k_off=4: 2x2x1 (Simple)
    
    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    add_simple_block(0, 4)
    
    add_strassen_block(2, 0)
    add_strassen_block(2, 2)
    add_simple_block(2, 4)
    
    if rank_offset != R:
        # Fallback or error
        pass
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
