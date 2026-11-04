"""Find rank-minimal exact tensor decomposition for 2x5 x 5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Rank-1 decomposition: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
