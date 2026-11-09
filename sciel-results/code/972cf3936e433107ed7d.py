"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Trivial decomposition: one rank-1 term for each (i, k, j) triplet
    # i in [0, n), k in [0, m), j in [0, p)
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the 1s
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # Calculate the rank index r
                # We map (i, k, j) to a unique r in [0, rank)
                # Order: i is slowest, then k, then j
                r = i * (m * p) + k * p + j
                
                # Index in U (A matrix, n x m)
                # Row-major: i * m + k
                u_idx = i * m + k
                
                # Index in V (B matrix, m x p)
                # Row-major: k * p + j
                v_idx = k * p + j
                
                # Index in W (C matrix, n x p)
                # Convention: W[k*n+i] where k is col (j) and i is row (i)
                # So w_idx = j * n + i
                w_idx = j * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
