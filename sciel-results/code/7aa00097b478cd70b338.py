"""Matrix multiplication 2x2x14: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """Return rank-56 schoolbook factor matrices for 2x2 by 2x14 matmul.
    
    Convention: U has shape (n*m, rank) = (4, rank)
               V has shape (m*p, rank) = (28, rank)
               W has shape (p*n, rank) = (28, rank) with index W[k*n+i, r]
    
    For each term (i, j, k): 
        U[i*m+j, r] = 1,  V[j*p+k, r] = 1,  W[k*n+i, r] = 1
    This ensures sum_r U[a,r]*V[b,r] reconstructs C[i,k] = sum_j A[i,j]*B[j,k]
    when a = i*m+j (input A index), b = j*p+k (input B index), output index = k*n+i.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
