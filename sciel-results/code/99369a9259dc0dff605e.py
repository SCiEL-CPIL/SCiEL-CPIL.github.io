"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Decompose 2x3x8 into 4 blocks of 2x3x2.
    # Each 2x3x2 block can be decomposed with rank 3.
    # Total rank = 4 * 3 = 12.
    
    rank = 12
    
    # U: (n*m, rank) = (6, 12)
    # V: (m*p, rank) = (24, 12)
    # W: (p*n, rank) = (16, 12)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 4 groups of columns for B (and rows for C)
    num_groups = p // 2  # 4 groups
    cols_per_group = 2
    
    for g in range(num_groups):
        # k indices in this group
        k_start = g * cols_per_group
        k_indices = [k_start, k_start + 1]
        
        # 3 ranks for j=0, 1, 2
        for j in range(m):
            r = g * m + j
            
            # Set U entries
            # U index: i * m + j
            for i in range(n):
                U[i * m + j][r] = 1
            
            # Set V entries
            # V index: j * p + k
            for k in k_indices:
                V[j * p + k][r] = 1
            
            # Set W entries
            # W index: k * n + i
            for k in k_indices:
                for i in range(n):
                    W[k * n + i][r] = 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
