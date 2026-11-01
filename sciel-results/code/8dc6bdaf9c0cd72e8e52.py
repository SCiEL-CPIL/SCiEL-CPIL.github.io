"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Split m=5 into m1=1, m2=4.
    # Part 1: 2x1x6. Rank = 12 (direct).
    # Part 2: 2x4x6. Split j=4 into two pairs, k=6 into three pairs.
    #   Each 2x2x2 block uses Strassen (rank 7).
    #   Total for Part 2: 2 * 3 * 7 = 42.
    # Total Rank = 54.
    
    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0, direct terms
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
    
    # Strassen terms for 2x2x2:
    # Each term is (u_comps, v_comps, w_comps)
    # u_comps: list of (local_i, local_j, val) for A[local_i, local_j]
    # v_comps: list of (local_j, local_k, val) for B[local_j, local_k]
    # w_comps: list of (local_i, local_k, val) for C[local_i, local_k]
    
    strassen_terms = [
        # T1: (A00+A11)(B00+B11) -> C00+1, C11+1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # T2: (A10+A11)B00 -> C10+1, C11-1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # T3: A00(B01-B11) -> C01+1, C11+1
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # T4: A11(B10-B00) -> C00+1, C10+1
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # T5: (A00+A01)B11 -> C00-1, C01+1
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # T6: (A10-A00)(B00+B01) -> C11+1
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # T7: (A01-A11)(B10+B11) -> C00+1
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    # Part 2: j in 1..4, split into two blocks of 2
    for block_j in range(2):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        j_base = 1 + block_j * 2
        # Split k=6 into three pairs
        for block_k in range(3):
            k_base = block_k * 2
            
            for term in strassen_terms:
                u_comps, v_comps, w_comps = term
                
                for (li, lj, val) in u_comps:
                    gi = li
                    gj = j_base + lj
                    U[gi * m + gj][col] += val
                
                for (lj, lk, val) in v_comps:
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col] += val
                
                for (li, lk, val) in w_comps:
                    gi = li
                    gk = k_base + lk
                    W[gk * n + gi][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
