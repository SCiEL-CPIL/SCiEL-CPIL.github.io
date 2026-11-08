import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start = time.time()
    deadline = start + 130.0
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition
    # Strategy: Use M(2,2,3) rank 11 blocks
    # M(3,4,6) = M(1,4,6) + M(2,4,6)
    # M(1,4,6): schoolbook = 24
    # M(2,4,6): split m into 2+2, each M(2,2,6) = 3*M(2,2,2) = 21
    # M(2,4,6) = 2*21 = 42
    # Total = 24 + 42 = 66
    
    # Alternative: M(3,4,6) split n=1+2, m=2+2, p=2+2+2
    # 8 blocks: 4x M(1,2,2) + 4x M(2,2,2) = 4*4 + 4*7 = 16+28 = 44? No...
    
    # Let me try: n=1+2, m=4, p=2+2+2
    # M(1,4,2) = 8, M(1,4,2) = 8, M(1,4,2) = 8 => 24
    # M(2,4,2) = 14, M(2,4,2) = 14, M(2,4,2) = 14 => 42
    # Total = 66
    
    # Try n=1+2, m=2+2, p=6
    # M(1,2,6) = 12, M(1,2,6) = 12 => 24
    # M(2,2,6) = 21, M(2,2,6) = 21 => 42
    # Total = 66
    
    # Try n=3, m=2+2, p=2+2+2
    # 6 blocks of M(3,2,2) = 3*4 + 2*7 = 12+14 = 26 each? No
    # M(3,2,2): n=1+2, so M(1,2,2)+M(2,2,2) = 4+7 = 11
    # 6 blocks * 11 = 66
    
    # It seems 66 is hard to beat with simple block decompositions.
    # Let me try a more careful construction.
    
    rank = 66
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
        for j in range(4):
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

    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    for block_k in range(3):
        if time.time() > deadline:
            break
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)

    return {"U": U, "V": V, "W": W}
