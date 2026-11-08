"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    # We'll build rank-13 decompositions for 4x4x4 and use them for the 4x4x7 case.
    # The 4x4x7 tensor T_{i,j,k} = delta_{i,j} (identity in k dimension for each pair)
    # Actually, the tensor for matrix multiplication (4x4)*(4x7) is:
    # T_{i,j,k} = 1 if j == k (the middle index connects), else 0
    # Wait, let me re-read the convention.
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (28, rank)
    # The tensor entry T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication C = A * B where A is 4x4 and B is 4x7:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T_{i,j,k} = 1 (the tensor is the identity tensor in the j dimension)
    
    # Let's use the Strassen-based decomposition for 4x4x4 and adapt for 4x4x7.
    # Actually, let's just use the schoolbook for now to be safe, but try to improve.
    # The known rank for 4x4x4 is 13 (Strassen).
    # For 4x4x7, we can think of it as 7 separate 4x4x1 multiplications, or as a single tensor.
    
    # Let's implement a simple but valid decomposition first, then try to optimize.
    # Schoolbook rank = 4*4*7 = 112
    
    # Let's try to use Strassen's 13-term formula for 4x4x4 and embed it.
    # The 4x4x7 tensor can be seen as having 7 "layers" in the k dimension.
    
    # Actually, the simplest correct approach: use the known rank-13 decomposition for 4x4x4.
    # For 4x4x7, we need rank at most 13*7/4 * something... 
    # Let me just implement the schoolbook to be safe, then we can evolve.
    
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
