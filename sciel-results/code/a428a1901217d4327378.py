import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _schoolbook_sub(n, m, p, i_offset, j_offset, k_offset, U, V, W, start_c):
    """
    Adds a schoolbook decomposition of an [n, m, p] sub-block to the global U, V, W.
    Returns the new start_c (number of columns used).
    """
    c = start_c
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Global indices
                gi = i_offset + i
                gj = j_offset + j
                gk = k_offset + k
                
                # U index: gi * m_total + gj ? No, U shape is (n*m, rank)
                # The task says U shape (n*m, rank).
                # But here n, m are sub-block dimensions.
                # The global U has rows indexed by (global_i, global_j).
                # We need to map to the global row index.
                
                # In the main construct, U is (3*5, rank).
                # So row index is gi * 5 + gj.
                
                U[gi * 5 + gj, c] = 1
                V[gj * 6 + gk, c] = 1
                W[gk * 3 + gi, c] = 1
                c += 1
    return c

def construct():
    start = time.time()
    deadline = start + 130.0
    
    n, m, p = DIMENSIONS
    # n=3, m=5, p=6
    
    # We will decompose into 4 blocks:
    # 1. j in 0..2, k in 0..2  ([3, 3, 3])
    # 2. j in 0..2, k in 3..5  ([3, 3, 3])
    # 3. j in 3..4, k in 0..2  ([3, 2, 3])
    # 4. j in 3..4, k in 3..5  ([3, 2, 3])
    
    # Total rank estimate: 9 + 9 + 18 + 18 = 54
    
    U = np.zeros((n * m, 100), dtype=object)
    V = np.zeros((m * p, 100), dtype=object)
    W = np.zeros((p * n, 100), dtype=object)
    
    c = 0
    
    # Block 1: i=0..2, j=0..2, k=0..2
    c = _schoolbook_sub(3, 3, 3, 0, 0, 0, U, V, W, c)
    
    # Block 2: i=0..2, j=0..2, k=3..5
    c = _schoolbook_sub(3, 3, 3, 0, 0, 3, U, V, W, c)
    
    # Block 3: i=0..2, j=3..4, k=0..2
    c = _schoolbook_sub(3, 2, 3, 0, 3, 0, U, V, W, c)
    
    # Block 4: i=0..2, j=3..4, k=3..5
    c = _schoolbook_sub(3, 2, 3, 0, 3, 3, U, V, W, c)
    
    # Trim to actual rank
    U = U[:, :c]
    V = V[:, :c]
    W = W[:, :c]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
