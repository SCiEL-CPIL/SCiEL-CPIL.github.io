import time

_START = time.time()
_DEADLINE = 140  # seconds

def construct():
    # 3x3 * 3x6 matrix multiplication
    # n=3 rows of A, m=3 cols of A / rows of B, p=6 cols of B
    # Output C is 3x6
    # Tensor T[i,j,k] = sum over appropriate indices
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (18, rank)
    # W shape: (p*n, rank) = (18, rank)
    # Convention: W[k*n+i, r]
    
    # The 3x3x6 matmul tensor is the direct sum of two 3x3x3 matmul tensors
    # since the output has 6 columns = 2 * 3 columns
    # So we can use two copies of a 3x3x3 decomposition
    # Schoolbook 3x3x3: rank 27, total rank 54
    # We need to find a better 3x3x3 decomposition if possible
    
    # For now, use schoolbook to ensure correctness
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # for each (i,j,k)
    
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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
