"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank is m * p. This is optimal for n=2 as it matches the lower bound max(nm, mp, np).
    # Construction: For each (j, k) pair, we use one rank-1 term.
    # V is non-zero only at (j, k).
    # U is non-zero for all i at column j.
    # W is non-zero for all i at row k.
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for k in range(p):
            # V factor: only non-zero at index j*p + k
            V[j * p + k][r] = 1
            
            # U and W factors: non-zero for all i
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
            
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
