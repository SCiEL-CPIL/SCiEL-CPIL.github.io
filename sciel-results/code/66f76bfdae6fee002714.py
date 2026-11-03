"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition strategy.
    # Split p=6 into 3 blocks of size 2.
    # For each block, we use a rank-6 decomposition of 3x3 by 3x2.
    # The rank-6 decomposition reuses the same U factors for each column in the block.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # There are 3 blocks (b=0, 1, 2)
    # Each block has 2 columns (k_local=0, 1)
    # For each column, we have 3 terms (j=0, 1, 2)
    for b in range(3):
        for k_local in range(2):
            k_global = b * 2 + k_local
            for j in range(3):
                # U factor: depends on j.
                # U[i, j, r] = 1 if j matches, else 0.
                # U is flattened: index i*m + j.
                for i in range(n):
                    U[i * m + j][col] = 1
                
                # V factor: depends on j and k_global.
                # V[j, k, r] = 1 if j and k match.
                # V is flattened: index j*p + k.
                V[j * p + k_global][col] = 1
                
                # W factor: depends on k_global.
                # W[k, i, r] = 1 if k matches.
                # W is flattened: index k*n + i.
                for i in range(n):
                    W[k_global * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
