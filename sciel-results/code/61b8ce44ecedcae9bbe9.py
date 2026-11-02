"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT over Z_2 x Z_7 for 2x4 * 4x7 matrix multiplication.
    
    Key insight: We can use the structure of the problem where n=2 is small.
    The group G = Z_2 x Z_7 has order 14.
    
    For matrix multiplication C = A*B where A is 2x4, B is 4x7:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We embed this into a convolution-like structure over G = Z_2 x Z_7.
    The idea is to use the Fourier transform over G to diagonalize the multiplication.
    
    However, for exact integer/half-integer decomposition, we need to be careful.
    Let's use a direct construction based on the structure.
    
    Actually, let's use a simpler but effective approach:
    Since n=2 is small, we can use the fact that we can combine the two rows
    of A using sum and difference to reduce the effective dimension.
    
    C[0,k] = sum_j A[0,j] * B[j,k]
    C[1,k] = sum_j A[1,j] * B[j,k]
    
    Let S[j] = A[0,j] + A[1,j] and D[j] = A[0,j] - A[1,j]
    Then:
    C[0,k] = (sum_j S[j]*B[j,k] + sum_j D[j]*B[j,k]) / 2
    C[1,k] = (sum_j S[j]*B[j,k] - sum_j D[j]*B[j,k]) / 2
    
    This doesn't reduce rank directly. Let me try a different approach.
    
    Using WFT over Z_2 x Z_7:
    We can think of the multiplication as a bilinear form. The rank of the
    multiplication tensor for (n x m) * (m x p) is at most n*m*p.
    
    For n=2, m=4, p=7, the schoolbook rank is 56.
    
    Let's try to use the structure of Z_2 x Z_7 more directly.
    The characters of Z_2 x Z_7 are indexed by (u,v) where u in {0,1}, v in {0,...,6}.
    
    For the exact decomposition, I'll construct it using the following approach:
    We can achieve rank 28 by exploiting the fact that m=4 and using a
    structured decomposition.
    
    Actually, let me just use the schoolbook decomposition which is guaranteed
    to be correct, with rank 56.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n * m * p = 56
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
