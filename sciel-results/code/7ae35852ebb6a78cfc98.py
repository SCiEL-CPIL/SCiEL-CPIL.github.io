"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = p (the all-ones tensor of size n x m x p has rank exactly p)
    rank = p  # = 5
    
    # U: shape (n*m, rank) = (8, 5), all ones
    U = [[1] * rank for _ in range(n * m)]
    
    # V: shape (m*p, rank) = (20, 5), V[j*p+k, r] = 1 if k==r else 0
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][k] = 1
    
    # W: shape (p*n, rank) = (10, 5), W[k*n+i, r] = 1 if k==r else 0
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            W[k * n + i][k] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
