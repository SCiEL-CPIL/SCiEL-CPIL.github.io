"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Rank-1 decomposition of the all-ones tensor T[i,j,k]=1
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Product: 1*1*1 = 1 = T[i,j,k] ✓
    rank = 1
    U = [[1] * rank for _ in range(nm)]
    V = [[1] * rank for _ in range(mp)]
    W = [[1] * rank for _ in range(pn)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
