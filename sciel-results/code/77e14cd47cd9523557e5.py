"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a Kronecker/block decomposition.
    # Split the 5 columns of A into: 1 + 4 (i.e., j=0 and j=1..4)
    # Split the 6 rows of B into: 2 + 2 + 2 (i.e., k=0,1; k=2,3; k=4,5)
    # For the j=0 part: 2x1x6 -> rank 12 (each of 2*6 products)
    # For the j=1..4 part: 2x4x6
    #   Split 4 columns into 2+2, and 6 rows into 2+2+2
    #   So we have 3 blocks of 2x2x2, each with Strassen rank 7
    #   Total for j=1..4: 3 * 7 = 21
    # Total rank = 12 + 21 = 33
    
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0, all k in 0..5
    # A[i,0] * B[0,k] for i in 0,1 and k in 0..5
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j in 1..4, split into 2 blocks of 2 columns each
    # Block 1: j = 1, 2
    # Block 2: j = 3, 4
    # For each block, split k into 3 groups of 2
    # Use Strassen for each 2x2x2 sub-block
    
    strassen_terms = [
        # (u_list, v_list, w_list) where each list contains (local_i, local_j, val)
        # T1: (a+d)(e+h) -> C00 += 1, C11 += 1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # T2: (c+d)e -> C10 += 1, C11 -= 1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # T3: a(f-h) -> C01 += 1, C11 += 1
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # T4: d(g-e) -> C00 += 1, C10 += 1
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # T5: (a+b)h -> C00 -= 1, C01 += 1
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # T6: (c-a)(e+f) -> C11 += 1
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # T7: (b-d)(g+h) -> C00 += 1
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    for block_j in range(2):
        j_base = 1 + block_j * 2
        for block_k in range(3):
            k_base = block_k * 2
            for term in strassen_terms:
                u_list, v_list, w_list = term
                
                for (li, lj, val) in u_list:
                    gi = li
                    gj = j_base + lj
                    U[gi * m + gj][col] += val
                
                for (lj, lk, val) in v_list:
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col] += val
                
                for (li, lk, val) in w_list:
                    gi = li
                    gk = k_base + lk
                    W[gk * n + gi][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
