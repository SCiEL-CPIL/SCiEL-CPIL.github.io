import numpy as np
import time

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140


def construct():
    n, m, p = DIMENSIONS
    # We will use a rank-63 decomposition by combining:
    # 3 blocks of M(1,4,2) [rank 8 each = 24]
    # 3 blocks of M(2,4,2) using Strassen-like rank 12 per block [36]
    # Total 24 + 36 = 60? Wait, M(2,4,2) can be done in rank 14 (2*Strassen).
    # Let's see if we can do better.
    # Actually, the standard bound for M(3,4,6) is n*m*p = 72.
    # Using Strassen on 2x2 blocks gives 66.
    # Let's try to find a decomposition of rank 60 or 63.
    
    # Known lower bound: 3*4*6 = 72.
    # Known upper bounds via Strassen:
    # M(3,4,6) = 3 * M(1,4,2) + 3 * M(2,4,2)
    # M(1,4,2) = 8 (schoolbook)
    # M(2,4,2) = 2 * M(2,2,2) = 14 (2*Strassen)
    # Total = 3*8 + 3*14 = 24 + 42 = 66.
    
    # Can we do M(2,4,2) in rank 12?
    # M(2,4,2) is equivalent to multiplying a 2x4 matrix by a 4x2 matrix.
    # This is a 2x2 matrix multiplication of size 2, but with intermediate dimension 4.
    # Actually, M(2,4,2) can be decomposed as:
    # A(2x4) * B(4x2)
    # Let A = [A1 A2] where A1, A2 are 2x2.
    # Let B = [B1; B2] where B1, B2 are 2x2.
    # C = A1*B1 + A2*B2.
    # This is a sum of two 2x2 matrix multiplications.
    # Each 2x2 matrix multiplication has rank 7 (Strassen).
    # So M(2,4,2) has rank at most 14.
    # Is there a way to share the Strassen coefficients?
    # If we use the same Strassen decomposition for both, we need 14 ranks.
    # If we can share, maybe less.
    
    # Let's try rank 63.
    # How?
    # Maybe M(3,4,6) can be done in rank 63?
    # 3 * M(1,4,2) + 3 * M(2,4,2) = 24 + 39 = 63?
    # Can M(2,4,2) be done in rank 13?
    
    # Let's stick to rank 66 for now, but use a more compact implementation.
    # Or maybe we can find a rank 60 decomposition?
    
    # Actually, let's use the known best upper bound.
    # According to literature, M(3,4,6) <= 66.
    # Let's see if we can do 63.
    
    # For now, let's implement rank 66 cleanly.
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
    
    def add_1_4_2(i_off, j_off, k_off):
        # n=1, m=4, p=2
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
        
        # Strassen 2x2x2
        m1 = lambda: [u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1]
        m2 = lambda: [u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1]
        m3 = lambda: [u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1]
        m4 = lambda: [u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1]
        m5 = lambda: [u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1]
        m6 = lambda: [u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1]
        m7 = lambda: [u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1]
        
        for func in [m1, m2, m3, m4, m5, m6, m7]:
            args = func()
            add_rank1(*args)
    
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
    
    return {"U": U, "V": V, "W": W}
