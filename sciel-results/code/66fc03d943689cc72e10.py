"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a Kronecker product based tensor decomposition for matrix multiplication.
    Decomposes 2x5x5 into (2x1x1) x (1x5x5).
    """
    n, m, p = DIMENSIONS
    
    # Sub-problem 1: 2x1x1
    n1, m1, p1 = 2, 1, 1
    # Sub-problem 2: 1x5x5
    n2, m2, p2 = 1, 5, 5
    
    # Verify dimensions match
    assert n == n1 * n2
    assert m == m1 * m2
    assert p == p1 * p2
    
    rank1 = n1 * m1 * p1 # 2
    rank2 = n2 * m2 * p2 # 25
    total_rank = rank1 * rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Iterate through all combinations of local terms
    for r1 in range(rank1):
        for r2 in range(rank2):
            r = r1 * rank2 + r2
            
            # Determine which global indices are active for this term
            # Term r1 in sub-problem 1 activates specific (i1, m1, p1) indices
            # Term r2 in sub-problem 2 activates specific (i2, m2, p2) indices
            
            # For schoolbook decomposition of sub-problem 1 (2x1x1):
            # r1 corresponds to (i1, j1, k1).
            # i1 = r1 // (m1*p1) = r1
            # j1 = (r1 % (m1*p1)) // p1 = 0
            # k1 = r1 % p1 = 0
            i1 = r1
            j1 = 0
            k1 = 0
            
            # For schoolbook decomposition of sub-problem 2 (1x5x5):
            # r2 corresponds to (i2, j2, k2).
            # i2 = r2 // (m2*p2) = 0
            # j2 = (r2 % (m2*p2)) // p2 = r2 // 5
            # k2 = r2 % p2 = r2 % 5
            i2 = 0
            j2 = r2 // p2
            k2 = r2 % p2
            
            # Map local indices to global indices
            # Global i = i1 * n2 + i2
            i = i1 * n2 + i2
            # Global j = j1 * m2 + j2
            j = j1 * m2 + j2
            # Global k = k1 * p2 + k2
            k = k1 * p2 + k2
            
            # Global indices for U, V, W
            # U index: i * m + j
            u_idx = i * m + j
            # V index: j * p + k
            v_idx = j * p + k
            # W index: k * n + i (Trace convention)
            w_idx = k * n + i
            
            U[u_idx][r] = 1
            V[v_idx][r] = 1
            W[w_idx][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
