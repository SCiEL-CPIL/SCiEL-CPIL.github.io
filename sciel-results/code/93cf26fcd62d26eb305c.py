"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Deterministic Kronecker/block decomposition:
    # Split m=5 into m1=1, m2=4
    # Part 1: 2x1x6 -> rank 12
    # Part 2: 2x4x6 -> split m=4 into 2+2 -> 2 * rank(2,2,6)
    # rank(2,2,6): split p=6 into 3 pairs of 2 -> 3 * rank(2,2,2) = 3*7 = 21
    # Total rank = 12 + 2*21 = 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0, naive 2x1x6
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j in {1,2,3,4}, split into two 2x2x6 blocks
    # Block 1: j in {1,2}, Block 2: j in {3,4}
    for block_idx in range(2):
        j_base = 1 + block_idx * 2  # maps local j' in {0,1} to global j
        
        # For each pair of k columns, apply Strassen 2x2x2
        for s in range(3):
            k_base = s * 2  # maps local k' in {0,1} to global k
            
            # Strassen 2x2x2 terms:
            # Local A indices: (i', j') with i',j' in {0,1}
            # Local B indices: (j', k') with j',k' in {0,1}
            # Local C indices: (i', k') with i',k' in {0,1}
            #
            # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
            # M2 = (A10+A11)*B00       -> C10 += M2, C11 -= M2
            # M3 = A00*(B01-B11)       -> C01 += M3, C11 += M3
            # M4 = A11*(B10-B00)       -> C00 += M4, C10 += M4
            # M5 = (A00+A01)*B11       -> C00 -= M5, C01 += M5
            # M6 = (A10-A00)*(B00+B01) -> C11 += M6
            # M7 = (A01-A11)*(B10+B11) -> C00 += M7
            
            strassen_terms = [
                # (u_terms, v_terms, w_terms)
                # Each term is list of (local_idx, val)
                # u: (i', j'), v: (j', k'), w: (i', k')
                (
                    [(0,0,1), (1,1,1)],
                    [(0,0,1), (1,1,1)],
                    [(0,0,1), (1,1,1)]
                ),
                (
                    [(1,0,1), (1,1,1)],
                    [(0,0,1)],
                    [(1,0,1), (1,1,-1)]
                ),
                (
                    [(0,0,1)],
                    [(0,1,1), (1,1,-1)],
                    [(0,1,1), (1,1,1)]
                ),
                (
                    [(1,1,1)],
                    [(1,0,1), (0,0,-1)],
                    [(0,0,1), (1,0,1)]
                ),
                (
                    [(0,0,1), (0,1,1)],
                    [(1,1,1)],
                    [(0,0,-1), (0,1,1)]
                ),
                (
                    [(1,0,1), (0,0,-1)],
                    [(0,0,1), (0,1,1)],
                    [(1,1,1)]
                ),
                (
                    [(0,1,1), (1,1,-1)],
                    [(1,0,1), (1,1,1)],
                    [(0,0,1)]
                ),
            ]
            
            for u_terms, v_terms, w_terms in strassen_terms:
                for (li, lj, val) in u_terms:
                    gi = li
                    gj = j_base + lj
                    U[gi * m + gj][col] += val
                
                for (lj, lk, val) in v_terms:
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col] += val
                
                for (li, lk, val) in w_terms:
                    gi = li
                    gk = k_base + lk
                    W[gk * n + gi][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
