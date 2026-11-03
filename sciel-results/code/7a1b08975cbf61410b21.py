"""Tensor decomposition for 2x5x6 matrix multiplication using a ladder/shifted approach."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    
    # Schoolbook rank is n*m*p = 60
    # We attempt a rank reduction using the ladder technique for n=2.
    
    # The idea: C[i,k] = sum_j A[i,j] B[j,k]
    # We can try to combine terms across j or i.
    
    # For n=2, we have:
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # A known technique for small n is to use "block" Strassen-like ideas.
    # However, for n=2, m=5, p=6, let's try to see if we can save rank.
    
    # Let's attempt a rank-59 decomposition by finding a linear dependency
    # or a clever combination. If not possible, return schoolbook.
    
    # Actually, let's try a specific construction:
    # We can write the tensor as a sum of rank-1 terms.
    # The standard schoolbook is 60 terms.
    
    # Let's try to use the fact that for some (j,k) pairs, we can combine
    # the contributions to C[0,k] and C[1,k] if A[0,j] and A[1,j] are
    # related, but they are independent variables, so we can't.
    
    # Alternative: try to find a rank-59 decomposition by solving a linear system.
    # This is computationally expensive but might be feasible for small dimensions.
    
    # For now, let's try a heuristic: can we replace two schoolbook terms with one?
    # Term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # If we have terms (0,j,k) and (1,j,k), they differ only in the W index.
    # We cannot combine them into a single rank-1 term because W[0] and W[1]
    # are different basis vectors.
    
    # Let's try a different approach: use the "ladder" decomposition.
    # For n=2, we can write:
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + ... + A[0,4]B[4,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + ... + A[1,4]B[4,k]
    
    # Consider the sums S_k = A[0,k]B[0,k] + A[1,k]B[1,k] (if m=p, but here m=5, p=6)
    # Since m != p, this doesn't directly apply.
    
    # Let's just return the schoolbook decomposition for now, as it's guaranteed correct.
    rank = n * m * p
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
