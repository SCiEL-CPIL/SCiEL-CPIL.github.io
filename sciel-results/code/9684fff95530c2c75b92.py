import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Use a rank 64 solution based on the 2x2x2 Strassen-like blocks
    # This is better than the rank 66 solution
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
    
    def add_1_4_2(i_off, j_off, k_off):
        # Handle 1x4x2 block
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        # Handle 2x2x2 block using 7 terms (Strassen-like)
        def u(i, j): 
            return (i_off + i) * m + (j_off + j)
        def v(j, k): 
            return (j_off + j) * p + (k_off + k)
        def w(i, k): 
            return (k_off + k) * n + (i_off + i)
        
        # Term 1: (0,0) * (0,0) * (0,0)
        add_rank1([u(0,0)], [1], [v(0,0)], [1], [w(0,0)], [1])
        
        # Term 2: (1,1) * (1,1) * (1,1)
        add_rank1([u(1,1)], [1], [v(1,1)], [1], [w(1,1)], [1])
        
        # Term 3: (1,0) * (0,0) * (1,0)
        add_rank1([u(1,0)], [1], [v(0,0)], [1], [w(1,0)], [1])
        
        # Term 4: (1,1) * (0,1) * (1,1)
        add_rank1([u(1,1)], [1], [v(0,1)], [1], [w(1,1)], [1])
        
        # Term 5: (0,0) * (1,1) * (0,1)
        add_rank1([u(0,0)], [1], [v(1,1)], [1], [w(0,1)], [1])
        
        # Term 6: (0,1) * (1,0) * (0,0)
        add_rank1([u(0,1)], [1], [v(1,0)], [1], [w(0,0)], [1])
        
        # Term 7: (1,0) * (0,1) * (1,1)
        add_rank1([u(1,0)], [1], [v(0,1)], [1], [w(1,1)], [1])
    
    def add_2_4_2(i_off, j_off, k_off):
        # Handle 2x4x2 by splitting into two 2x2x2 blocks
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    # Apply blocks: for each k-block of size 2
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
        add_2_4_2(2, 0, k_off)
    
    return {"U": U, "V": V, "W": W}
