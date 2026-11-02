import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    # Total rank 56
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Local Strassen decomposition for 2x2x2
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: 2*i + j
    # V index: 2*j + k
    # W index: 2*k + i
    
    # Define the 7 factors as lists of (index, value)
    # M1
    u0 = [(0, 1), (3, 1)]
    v0 = [(0, 1), (3, 1)]
    w0 = [(0, 1), (3, 1)] # c00, c11
    
    # M2
    u1 = [(2, 1), (3, 1)]
    v1 = [(0, 1)]
    w1 = [(1, 1), (3, -1)] # c10, c11
    
    # M3
    u2 = [(0, 1)]
    v2 = [(2, 1), (3, -1)]
    w2 = [(2, 1), (3, 1)] # c01, c11
    
    # M4
    u3 = [(3, 1)]
    v3 = [(1, 1), (0, -1)]
    w3 = [(0, 1), (1, 1)] # c00, c10
    
    # M5
    u4 = [(0, 1), (1, 1)]
    v4 = [(3, 1)]
    w4 = [(0, -1), (2, 1)] # c00, c01
    
    # M6
    u5 = [(2, 1), (0, -1)]
    v5 = [(0, 1), (2, 1)]
    w5 = [(3, 1)] # c11
    
    # M7
    u6 = [(1, 1), (3, -1)]
    v6 = [(1, 1), (3, 1)]
    w6 = [(0, 1)] # c00
    
    local_factors = [
        (u0, v0, w0),
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6)
    ]
    
    # 8 sub-problems
    # s = 4*src + q
    # src: 0 for A_L/B_T (j in 0,1), 1 for A_R/B_B (j in 2,3)
    # q: 0..3 for k block (k in 2q, 2q+1)
    
    for s in range(8):
        src = s // 4
        q = s % 4
        
        # j offset
        j_offset = 2 * src
        # k offset
        k_offset = 2 * q
        
        for r in range(7):
            rank_idx = 7 * s + r
            u_loc, v_loc, w_loc = local_factors[r]
            
            # Update U
            for idx_loc, val in u_loc:
                i_loc = idx_loc // 2
                j_loc = idx_loc % 2
                i_glob = i_loc
                j_glob = j_loc + j_offset
                u_idx = i_glob * m + j_glob
                U[u_idx][rank_idx] += val
                
            # Update V
            for idx_loc, val in v_loc:
                j_loc = idx_loc // 2
                k_loc = idx_loc % 2
                j_glob = j_loc + j_offset
                k_glob = k_loc + k_offset
                v_idx = j_glob * p + k_glob
                V[v_idx][rank_idx] += val
                
            # Update W
            for idx_loc, val in w_loc:
                k_loc = idx_loc // 2
                i_loc = idx_loc % 2
                k_glob = k_loc + k_offset
                i_glob = i_loc
                w_idx = k_glob * n + i_glob
                W[w_idx][rank_idx] += val
                
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
