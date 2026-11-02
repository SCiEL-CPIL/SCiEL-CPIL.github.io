"""Program for rank-minimal tensor decomposition of 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    
    The multiplication tensor T_{2,5,5} has entries T[i,j,k] = 1 where:
    - i in [0,2) is the row of A (and output C)
    - j in [0,5) is the column of A / row of B
    - k in [0,5) is the column of B / column of C
    
    The bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We use the schoolbook decomposition with rank = 2*5*5 = 50.
    Each rank-1 term corresponds to one (i,j,k) triple.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = n * m * p  # 50
    
    # U: shape (n*m, rank) = (10, 50) - indices A[i*m + j]
    # V: shape (m*p, rank) = (25, 50) - indices B[j*p + k]
    # W: shape (p*n, rank) = (10, 50) - indices C[k*n + i] per trace convention
    
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
