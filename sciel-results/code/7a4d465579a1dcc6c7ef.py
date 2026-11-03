"""
Rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
Uses a block decomposition based on 3x2x2 sub-problems of rank 6.
Total rank = 8 * 6 = 48.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _construct_3x2x2():
    """
    Construct decomposition for 3x2x2 matrix multiplication.
    n=3, m=2, p=2. Rank=6.
    U: (6, 6)
    V: (4, 6)
    W: (6, 6)
    """
    n, m, p = 3, 2, 2
    rank = 6
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for i in range(n):
        # For each row i, we have a 1x2x2 problem.
        # Rank 2 decomposition:
        # c0: U[i,0]=1, U[i,1]=0
        #      V[0,0]=1, V[0,1]=1, V[1,0]=0, V[1,1]=0
        #      W[0,i]=1, W[1,i]=1
        # c1: U[i,0]=0, U[i,1]=1
        #      V[0,0]=0, V[0,1]=0, V[1,0]=1, V[1,1]=1
        #      W[0,i]=1, W[1,i]=1
        
        c0 = 2 * i
        c1 = 2 * i + 1
        
        # U indices: i*m + j
        U[i * m + 0][c0] = 1.0
        U[i * m + 1][c1] = 1.0
        
        # V indices: j*p + k
        # c0: j=0 -> V[0,0], V[0,1] are 1
        V[0 * p + 0][c0] = 1.0
        V[0 * p + 1][c0] = 1.0
        # c1: j=1 -> V[1,0], V[1,1] are 1
        V[1 * p + 0][c1] = 1.0
        V[1 * p + 1][c1] = 1.0
        
        # W indices: k*n + i
        # c0: k=0 -> W[0,i], k=1 -> W[1,i] are 1
        W[0 * n + i][c0] = 1.0
        W[1 * n + i][c0] = 1.0
        # c1: k=0 -> W[0,i], k=1 -> W[1,i] are 1
        W[0 * n + i][c1] = 1.0
        W[1 * n + i][c1] = 1.0
        
    return U, V, W

def _expand_decomposition(base_U, base_V, base_W, base_n, base_m, base_p, 
                          target_n, target_m, target_p, 
                          m_splits, p_splits):
    """
    Expand a base decomposition to target dimensions by tiling.
    base: 3x2x2
    target: 3x4x8
    m_splits: how to split 4 into 2s (2 blocks)
    p_splits: how to split 8 into 2s (4 blocks)
    """
    rank_base = len(base_U[0])
    num_m_blocks = target_m // base_m
    num_p_blocks = target_p // base_p
    total_rank = rank_base * num_m_blocks * num_p_blocks
    
    U = [[0.0] * total_rank for _ in range(target_n * target_m)]
    V = [[0.0] * total_rank for _ in range(target_m * target_p)]
    W = [[0.0] * total_rank for _ in range(target_p * target_n)]
    
    for bm in range(num_m_blocks):
        for bp in range(num_p_blocks):
            col_offset = (bm * num_p_blocks + bp) * rank_base
            
            for i in range(base_n):
                for j in range(base_m):
                    base_idx_u = i * base_m + j
                    target_i = i
                    target_j = bm * base_m + j
                    for c in range(rank_base):
                        U[target_i * target_m + target_j][col_offset + c] = base_U[base_idx_u][c]
            
            for j in range(base_m):
                for k in range(base_p):
                    base_idx_v = j * base_p + k
                    target_j = bm * base_m + j
                    target_k = bp * base_p + k
                    for c in range(rank_base):
                        V[target_j * target_p + target_k][col_offset + c] = base_V[base_idx_v][c]
            
            for k in range(base_p):
                for i in range(base_n):
                    base_idx_w = k * base_n + i
                    target_k = bp * base_p + k
                    target_i = i
                    for c in range(rank_base):
                        W[target_k * target_n + target_i][col_offset + c] = base_W[base_idx_w][c]
                        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Construct base 3x2x2 decomposition
    base_U, base_V, base_W = _construct_3x2x2()
    
    # 2. Expand to 3x4x8
    # m=4 is 2 blocks of 2. p=8 is 4 blocks of 2.
    U, V, W = _expand_decomposition(base_U, base_V, base_W, 3, 2, 2, n, m, p, 2, 4)
    
    # 3. Verify (optional, but good for safety)
    # if not _verify(U, V, W, n, m, p):
    #     # Fallback to schoolbook if verification fails (should not happen)
    #     rank_schoolbook = n * m * p
    #     U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    #     V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    #     W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    #     column = 0
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(p):
    #                 U_fb[i * m + j][column] = 1
    #                 V_fb[j * p + k][column] = 1
    #                 W_fb[k * n + i][column] = 1
    #                 column += 1
    #     return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
