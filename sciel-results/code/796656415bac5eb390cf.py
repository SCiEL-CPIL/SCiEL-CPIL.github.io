import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 3, 5, 7
    
    # Partitions
    # n: 2, 1
    # m: 2, 2, 1
    # p: 2, 2, 3
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 3]
    
    # Calculate global indices for each part
    n_indices = []
    start = 0
    for size in n_parts:
        n_indices.append(list(range(start, start + size)))
        start += size
        
    m_indices = []
    start = 0
    for size in m_parts:
        m_indices.append(list(range(start, start + size)))
        start += size
        
    p_indices = []
    start = 0
    for size in p_parts:
        p_indices.append(list(range(start, start + size)))
        start += size

    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    def add_block(n_idx, m_idx, p_idx, rank, u_local, v_local, w_local):
        """
        n_idx, m_idx, p_idx: lists of global indices
        u_local: matrix of shape (len(n_idx)*len(m_idx), rank)
        v_local: matrix of shape (len(m_idx)*len(p_idx), rank)
        w_local: matrix of shape (len(p_idx)*len(n_idx), rank)
        """
        # Map local row/col indices to global
        # Local A index: i * len(m_idx) + j -> Global: n_idx[i] * m + m_idx[j]
        # Local B index: j * len(p_idx) + k -> Global: m_idx[j] * p + p_idx[k]
        # Local C index: k * len(n_idx) + i -> Global: p_idx[k] * n + n_idx[i]
        
        n_size = len(n_idx)
        m_size = len(m_idx)
        p_size = len(p_idx)
        
        U_block = np.zeros((n * m, rank), dtype=np.int64)
        V_block = np.zeros((m * p, rank), dtype=np.int64)
        W_block = np.zeros((p * n, rank), dtype=np.int64)
        
        for r in range(rank):
            # U
            for i in range(n_size):
                for j in range(m_size):
                    val = u_local[i * m_size + j, r]
                    if val != 0:
                        U_block[n_idx[i] * m + m_idx[j], r] = val
            
            # V
            for j in range(m_size):
                for k in range(p_size):
                    val = v_local[j * p_size + k, r]
                    if val != 0:
                        V_block[m_idx[j] * p + p_idx[k], r] = val
            
            # W
            for k in range(p_size):
                for i in range(n_size):
                    val = w_local[k * n_size + i, r]
                    if val != 0:
                        W_block[p_idx[k] * n + n_idx[i], r] = val
        
        U_list.append(U_block)
        V_list.append(V_block)
        W_list.append(W_block)
        
    # Helper for schoolbook block
    def schoolbook_block(n_size, m_size, p_size):
        rank = n_size * m_size * p_size
        u = np.zeros((n_size * m_size, rank), dtype=np.int64)
        v = np.zeros((m_size * p_size, rank), dtype=np.int64)
        w = np.zeros((p_size * n_size, rank), dtype=np.int64)
        col = 0
        for i in range(n_size):
            for j in range(m_size):
                for k in range(p_size):
                    u[i * m_size + j, col] = 1
                    v[j * p_size + k, col] = 1
                    w[k * n_size + i, col] = 1
                    col += 1
        return rank, u, v, w

    # Helper for Strassen block (2x2x2)
    def strassen_block():
        rank = 7
        # U: 4x7
        u = np.array([
            [1, 1, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 1, 0, -1],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, -1, 0, 0, 1, 1]
        ], dtype=np.int64)
        
        # V: 4x7
        v = np.array([
            [1, 1, 0, -1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 1, -1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0]
        ], dtype=np.int64)
        
        # W: 4x7
        w = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=np.int64)
        
        return rank, u, v, w

    # Iterate over all blocks
    for i, n_idx in enumerate(n_indices):
        for j, m_idx in enumerate(m_indices):
            for k, p_idx in enumerate(p_indices):
                n_size = len(n_idx)
                m_size = len(m_idx)
                p_size = len(p_idx)
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    rank, u, v, w = strassen_block()
                else:
                    rank, u, v, w = schoolbook_block(n_size, m_size, p_size)
                
                add_block(n_idx, m_idx, p_idx, rank, u, v, w)
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Combine blocks
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
