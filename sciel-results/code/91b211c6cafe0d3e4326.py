import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose m=4 into two blocks of size 2.
    # For each block:
    #   Split n=3 into n1=2 and n2=1.
    #   Sub-block 1: n=2, m=2, p=7. Rank = 14.
    #   Sub-block 2: n=1, m=2, p=7. Rank = 1.
    # Total Rank = 2 * (14 + 1) = 30.
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += val
        for idx in v_indices:
            V[idx][col] += val
        for idx in w_indices:
            W[idx][col] += val
        col += 1

    # Iterate over m-blocks (0 and 1)
    for block in range(2):
        j_off = block * 2
        
        # Sub-block 1: n=2 (i=0,1), m=2 (j=j_off, j_off+1), p=7
        # Rank 14: 2 per column k
        for k in range(p):
            # Term 1: Contributes to i=0
            # U[0*m + j_off] = 1, U[0*m + j_off+1] = 1
            # V[(j_off)*p + k] = 1, V[(j_off+1)*p + k] = 1
            # W[k*n + 0] = 1
            u1 = [0 * m + j_off, 0 * m + j_off + 1]
            v1 = [j_off * p + k, (j_off + 1) * p + k]
            w1 = [k * n + 0]
            add_term(u1, v1, w1)
            
            # Term 2: Contributes to i=1
            # U[1*m + j_off] = 1, U[1*m + j_off+1] = 1
            # V[(j_off)*p + k] = 1, V[(j_off+1)*p + k] = 1
            # W[k*n + 1] = 1
            u2 = [1 * m + j_off, 1 * m + j_off + 1]
            v2 = [j_off * p + k, (j_off + 1) * p + k]
            w2 = [k * n + 1]
            add_term(u2, v2, w2)
            
        # Sub-block 2: n=1 (i=2), m=2 (j=j_off, j_off+1), p=7
        # Rank 1: 1 term total
        # U[2*m + j_off] = 1, U[2*m + j_off+1] = 1
        # V[(j_off)*p + k] = 1, V[(j_off+1)*p + k] = 1 for all k
        # W[k*n + 2] = 1 for all k
        u3 = [2 * m + j_off, 2 * m + j_off + 1]
        v3 = [j_off * p + k, (j_off + 1) * p + k] # This needs to be all k
        # Wait, add_term takes a list.
        # We need V entries for ALL k in this single term.
        v3_full = []
        for k in range(p):
            v3_full.append(j_off * p + k)
            v3_full.append((j_off + 1) * p + k)
        w3_full = []
        for k in range(p):
            w3_full.append(k * n + 2)
        
        add_term(u3, v3_full, w3_full)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
