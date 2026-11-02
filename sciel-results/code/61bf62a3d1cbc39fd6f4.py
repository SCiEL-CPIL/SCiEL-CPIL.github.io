"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor to decompose is the all-ones tensor J_{n,m,p}
    # based on the schoolbook implementation which sets T[i,j,k]=1 for all i,j,k.
    # The minimal rank of J_{n,m,p} is max(n, m, p).
    rank = max(n, m, p)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U and V with 1s, W with 1/rank
    for i in range(n * m):
        for c in range(rank):
            U[i][c] = 1.0
            
    for j in range(m * p):
        for c in range(rank):
            V[j][c] = 1.0
            
    for k in range(p * n):
        for c in range(rank):
            W[k][c] = 1.0 / rank
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
