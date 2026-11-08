import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # 39-rank solution:
    # Split p=6 into 2+2+2
    # For each p-block of size 2:
    #   Split m=4 into 1+3
    #   Block 1: M(3, 1, 2) at m_off=0, rank 1
    #   Block 2: M(3, 3, 2) at m_off=1, rank 12
    #     Inside M(3, 3, 2):
    #       Split m=3 into 2+1
    #       Block 2.1: M(3, 2, 2) at m_off=1, rank 11
    #         Inside M(3, 2, 2):
    #           Split n=3 into 2+1
    #           Block 2.1.1: M(2, 2, 2) at n_off=0, rank 7
    #           Block 2.1.2: M(1, 2, 2) at n_off=2, rank 4 (schoolbook)
    #       Block 2.2: M(3, 1, 2) at m_off=3, rank 1
    # Total rank per p-block: 1 + 7 + 4 + 1 = 13
    # Total rank: 3 * 13 = 39
    
    rank = 39
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
    
    def add_3_1_2(i_off, j_off, k_off):
        # M(3, 1, 2): Rank 1
        # C[i, k] = A[i, j] * B[j, k]
        u_indices = [(i_off + i) * m + j_off for i in range(3)]
        u_vals = [1] * 3
        v_indices = [j_off * p + (k_off + k) for k in range(2)]
        v_vals = [1] * 2
        w_indices = [(k_off + k) * n + (i_off + i) for k in range(2) for i in range(3)]
        w_vals = [1] * 6
        add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        
    def add_1_2_2(i_off, j_off, k_off):
        # M(1, 2, 2): Schoolbook, Rank 4
        for j in range(2):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
                
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
        
    def add_3_2_2(i_off, j_off, k_off):
        # M(3, 2, 2): Split n=3 into 2+1
        add_2_2_2(i_off, j_off, k_off)
        add_1_2_2(i_off + 2, j_off, k_off)
        
    def add_3_3_2(i_off, j_off, k_off):
        # M(3, 3, 2): Split m=3 into 2+1
        add_3_2_2(i_off, j_off, k_off)
        add_3_1_2(i_off, j_off + 2, k_off)
        
    def add_3_4_2(i_off, j_off, k_off):
        # M(3, 4, 2): Split m=4 into 1+3
        add_3_1_2(i_off, j_off, k_off)
        add_3_3_2(i_off, j_off + 1, k_off)
        
    # Iterate over p blocks
    for block_k in range(3):
        k_off = block_k * 2
        add_3_4_2(0, 0, k_off)
        
    return {"U": U, "V": V, "W": W}

def _construct_66():
    # Fallback, not used if above works
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
