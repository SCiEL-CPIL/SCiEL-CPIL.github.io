"""Program for 6x6 matrix multiplication with schoolbook (naive) decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition rank is n^3 for n x n matrices.
    # Here n=m=p=6, so rank = 6^3 = 216.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The schoolbook decomposition is the sum of outer products of the
    # individual entries.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Each term (i, j, k) contributes 1 to U[i, j], V[j, k], W[k, i].
    # We map the triple (i, j, k) to a rank index c = i * (m * p) + j * p + k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                
                # U index: row i, col j -> i * m + j
                U[i * m + j][c] = 1
                
                # V index: row j, col k -> j * p + k
                V[j * p + k][c] = 1
                
                # W index: row k, col i -> k * n + i (Trace convention W[k*n+i, r])
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
