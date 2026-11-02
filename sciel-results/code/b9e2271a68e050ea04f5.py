"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # Rank is n * p = 3 * 8 = 24
    rank = n * p
    
    # U shape: (n*m, rank) -> (9, 24)
    # V shape: (m*p, rank) -> (24, 24)
    # W shape: (p*n, rank) -> (24, 24)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index mapping:
    # U index a = i * m + j
    # V index b = j * p + k
    # W index c = k * n + i_out
    
    # Decomposition index r corresponds to (k, r_idx)
    # where k in 0..p-1 and r_idx in 0..n-1
    # Total rank = p * n
    
    col = 0
    for k in range(p):
        for r_idx in range(n):
            # Set U
            # U[i, j, (k, r_idx)] = 1 if i == r_idx
            for i in range(n):
                if i == r_idx:
                    for j in range(m):
                        a = i * m + j
                        U[a][col] = 1
                else:
                    pass # already 0
            
            # Set V
            # V[j, k', (k, r_idx)] = 1 if k' == k
            for j in range(m):
                for k_prime in range(p):
                    if k_prime == k:
                        b = j * p + k_prime
                        V[b][col] = 1
            
            # Set W
            # W[k_out, i_out, (k, r_idx)] = 1 if k_out == k and i_out == r_idx
            for k_out in range(p):
                for i_out in range(n):
                    if k_out == k and i_out == r_idx:
                        c = k_out * n + i_out
                        W[c][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
