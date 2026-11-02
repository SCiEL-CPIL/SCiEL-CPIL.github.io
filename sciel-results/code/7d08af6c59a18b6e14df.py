import time
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Split m=5 into m1=1 (direct) and m2=4 (Strassen+Direct per 2x4x2 block)
    # Part 1: 2x1x6 direct, rank = 1*6 = 6? No, n*p = 2*6 = 12.
    # Part 2: 2x4x6, split k=6 into 3 blocks of 2.
    #   Each 2x4x2: Strassen on (i=0,1; j=0,1; k=0,1) rank 7 + direct on (i=0,1; j=2,3; k=0,1) rank 4 = 11.
    #   Total Part 2: 3 * 11 = 33.
    # Grand total: 12 + 33 = 45.
    
    total_rank = 45
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0, direct 2x1x6
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Strassen 2x2x2 terms: (u_comps, v_comps, w_comps)
    # Each comp: (local_i, local_j, val) for U; (local_j, local_k, val) for V; (local_i, local_k, val) for W
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    # Part 2: j in 1..4, split k=6 into three blocks of 2
    for block_k in range(3):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        k_base = block_k * 2
        
        # Sub-block A: j=1,2 (local j=0,1), Strassen 2x2x2
        j_base_strassen = 1
        for term in strassen_terms:
            u_comps, v_comps, w_comps = term
            for (li, lj, val) in u_comps:
                gi = li
                gj = j_base_strassen + lj
                U[gi * m + gj][col] += val
            for (lj, lk, val) in v_comps:
                gj = j_base_strassen + lj
                gk = k_base + lk
                V[gj * p + gk][col] += val
            for (li, lk, val) in w_comps:
                gi = li
                gk = k_base + lk
                W[gk * n + gi][col] += val
            col += 1
        
        # Sub-block B: j=3,4 (local j=0,1), direct 2x2x2 (rank 4)
        j_base_direct = 3
        for li in range(n):
            for lj in range(2):
                for lk in range(2):
                    gi = li
                    gj = j_base_direct + lj
                    gk = k_base + lk
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
