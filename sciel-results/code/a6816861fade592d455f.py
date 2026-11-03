"""Optimized matrix multiplication 3x5x7 via block decomposition."""
import numpy as np

DIMENSIONS = [3, 5, 7]

def _schoolbook_decomp(n, m, p):
    """Generate schoolbook decomposition for n x m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 1 and 4
    # Split p=7 into 3 and 4
    # Blocks:
    # 1. n=3, m1=1, p1=3
    # 2. n=3, m1=1, p2=4
    # 3. n=3, m2=4, p1=3
    # 4. n=3, m2=4, p2=4
    
    blocks = [
        (0, 1, 0, 3),   # i_start, i_end, j_start, j_end, k_start, k_end? 
        # Let's define by indices:
        # Block 1: i in [0,3), j in [0,1), k in [0,3)
        # Block 2: i in [0,3), j in [0,1), k in [3,7)
        # Block 3: i in [0,3), j in [1,5), k in [0,3)
        # Block 4: i in [0,3), j in [1,5), k in [3,7)
    ]
    
    # Define blocks as (i_start, i_end, j_start, j_end, k_start, k_end)
    block_defs = [
        (0, 3, 0, 1, 0, 3),
        (0, 3, 0, 1, 3, 7),
        (0, 3, 1, 5, 0, 3),
        (0, 3, 1, 5, 3, 7)
    ]
    
    # Calculate ranks
    ranks = []
    for (i_s, i_e, j_s, j_e, k_s, k_e) in block_defs:
        ni = i_e - i_s
        mj = j_e - j_s
        pk = k_e - k_s
        ranks.append(ni * mj * pk)
        
    total_rank = sum(ranks)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_col = 0
    for (i_s, i_e, j_s, j_e, k_s, k_e), rank in zip(block_defs, ranks):
        ni = i_e - i_s
        mj = j_e - j_s
        pk = k_e - k_s
        
        # Get local decomposition
        U_loc, V_loc, W_loc = _schoolbook_decomp(ni, mj, pk)
        
        # Embed into global matrices
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        for r in range(rank):
            # U
            for i in range(ni):
                for j in range(mj):
                    val = U_loc[i * mj + j, r]
                    if val != 0:
                        U[(i_s + i) * m + (j_s + j), current_col + r] = val
            # V
            for j in range(mj):
                for k in range(pk):
                    val = V_loc[j * pk + k, r]
                    if val != 0:
                        V[(j_s + j) * p + (k_s + k), current_col + r] = val
            # W
            for k in range(pk):
                for i in range(ni):
                    val = W_loc[k * ni + i, r]
                    if val != 0:
                        W[(k_s + k) * n + (i_s + i), current_col + r] = val
                        
        current_col += rank
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
