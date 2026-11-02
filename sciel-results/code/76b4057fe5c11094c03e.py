import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    rank = 53
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen components for 2x2x2
    # Indices: U is 4x7, V is 4x7, W is 4x7
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook components for 2x2x1
    U_b = np.eye(4, 4)
    V_b = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_b = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # Assembly
    # 7 Strassen blocks for k=0..13
    col = 0
    for block_idx in range(7):
        k_start = block_idx * 2
        
        # U part: same for all blocks, just repeat
        # U is (n*m) x rank. For each block, U columns are identical to U_s.
        U[:, col:col+7] = U_s
        
        # V part: V is (m*p) x rank.
        # V_s is 4x7. Rows of V_s correspond to (j, k) indices for a 2x2 block.
        # j in 0..1, k in 0..1 relative to block.
        # Global k = k_start + k_rel.
        # V global row index: j * p + k_global.
        # V_s row index: j * 2 + k_rel.
        for j in range(2):
            for k_rel in range(2):
                v_row = j * p + (k_start + k_rel)
                v_s_row = j * 2 + k_rel
                V[v_row, col:col+7] = V_s[v_s_row, :]
                
        # W part: W is (p*n) x rank.
        # W_s is 4x7. Rows of W_s correspond to (k, i) indices.
        # k in 0..1 relative, i in 0..1.
        # Global k = k_start + k_rel.
        # W global row index: k_global * n + i.
        # W_s row index: k_rel * 2 + i.
        for k_rel in range(2):
            for i in range(2):
                w_row = (k_start + k_rel) * n + i
                w_s_row = k_rel * 2 + i
                W[w_row, col:col+7] = W_s[w_s_row, :]
                
        col += 7
        
    # 1 Schoolbook block for k=14
    k_start = 14
    U[:, col:col+4] = U_b
    
    # V part for schoolbook
    # V_b is 2x4. Rows correspond to j=0,1. k_rel=0.
    # Global k = 14.
    for j in range(2):
        v_row = j * p + 14
        V[v_row, col:col+4] = V_b[j, :]
        
    # W part for schoolbook
    # W_b is 2x4. Rows correspond to i=0,1. k_rel=0.
    # Global k = 14.
    for i in range(2):
        w_row = 14 * n + i
        W[w_row, col:col+4] = W_b[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
