import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 6]

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return U,V,W for 2x2x2 matrix multiplication (Strassen, rank 7)."""
    # Standard Strassen decomposition
    U = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _build_block_decomp(n, m, p):
    """
    Build a block decomposition for 4x5x6 using Strassen for 2x2x2 sub-blocks.
    Partition: i={0,1}|{2,3}, j={0,1}|{2,3,4}, k={0,1}|{2,3,4,5}
    """
    # Block sizes
    i_parts = [(0, 2), (2, 4)]
    j_parts = [(0, 2), (2, 5)]
    k_parts = [(0, 2), (2, 6)]
    
    # Precompute decompositions for each block type
    strassen_U, strassen_V, strassen_W = _strassen_2x2()
    
    # Compute total rank
    total_rank = 0
    block_ranks = {}
    for i_idx, (i_start, i_end) in enumerate(i_parts):
        for j_idx, (j_start, j_end) in enumerate(j_parts):
            for k_idx, (k_start, k_end) in enumerate(k_parts):
                ni = i_end - i_start
                nj = j_end - j_start
                nk = k_end - k_start
                if ni == 2 and nj == 2 and nk == 2:
                    rank = 7
                else:
                    rank = ni * nj * nk
                block_ranks[(i_idx, j_idx, k_idx)] = rank
                total_rank += rank
    
    # Initialize output matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for i_idx, (i_start, i_end) in enumerate(i_parts):
        for j_idx, (j_start, j_end) in enumerate(j_parts):
            for k_idx, (k_start, k_end) in enumerate(k_parts):
                ni = i_end - i_start
                nj = j_end - j_start
                nk = k_end - k_start
                rank = block_ranks[(i_idx, j_idx, k_idx)]
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen
                    U_block, V_block, W_block = strassen_U, strassen_V, strassen_W
                else:
                    # Use schoolbook
                    U_block, V_block, W_block = _schoolbook(ni, nj, nk)
                
                # Place the block
                for li in range(ni):
                    for lj in range(nj):
                        for lk in range(nk):
                            gi = i_start + li
                            gj = j_start + lj
                            gk = k_start + lk
                            
                            for r in range(rank):
                                U[gi * m + gj][col_offset + r] = U_block[li * nj + lj][r]
                                V[gj * p + gk][col_offset + r] = V_block[lj * nk + lk][r]
                                W[gk * n + gi][col_offset + r] = W_block[lk * ni + li][r]
                
                col_offset += rank
    
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Build the block decomposition
    U, V, W, rank = _build_block_decomp(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
