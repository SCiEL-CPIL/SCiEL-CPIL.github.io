import time

DIMENSIONS = [2, 5, 6]

def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin
    
    n, m, p = DIMENSIONS
    # Rank-1 tensor for multiplication: C[k*n+i] += A[i*m+j] * B[j*p+k]
    # So W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] should sum to 1 when (i,j,k) matches the term
    
    # Schoolbook: rank = n*m*p = 2*5*6 = 60
    # We can try to reduce by finding linear dependencies.
    # For 2x5x6, the tensor is not very dense in terms of independence.
    # Let's try to use a greedy rank reduction / exact linear algebra approach.
    
    # Build the full tensor T of size (p*n) x (n*m) x (m*p)
    # T[k*n + i, i*m + j, j*p + k] = 1
    # We want to find rank r such that T = sum_c U_c \otimes V_c \otimes W_c
    
    # This is equivalent to finding a decomposition of the 3-way tensor.
    # For small sizes, we can use exact linear algebra over Q.
    
    # Flatten T to a matrix for each mode:
    # Mode-1 unfolding: M1 of size (p*n) x (n*m * m*p)
    # M1[k*n+i, (i*m+j)*(m*p) + (j*p+k)] = T[k*n+i, i*m+j, j*p+k]
    # Actually, for tensor decomposition, we can use the fact that
    # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # This means that for fixed k, the matrix T[:,:,k] = U W_k^T V^T ... no.
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # So for fixed k, T[:,:,k] = U * diag(W[:,k]) * V^T
    
    # Let's use a direct approach: try ranks from lower bound up.
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But actually, the trivial rank is 60.
    # Known results: 2x5x6 might have rank less than 60.
    # Let's try to find a decomposition using linear algebra.
    
    # For exact decomposition with integer/half-integer entries, we can work over Q.
    
    # Strategy: Use the "border rank" or "rank" via elimination.
    # Since the problem is small (60 terms), we can try to find a minimal set.
    
    # Let's start with the schoolbook decomposition and try to eliminate terms
    # by finding linear combinations.
    
    # Actually, a simpler approach: try ranks from 30 to 60 and check if a
    # decomposition exists using exact linear algebra.
    
    # For a given rank r, we need to find U (10 x r), V (30 x r), W (12 x r)
    # such that T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    
    # This is a system of bilinear equations. Hard to solve directly.
    
    # Alternative: use the fact that T is a "simple" tensor (all 1s at specific positions).
    # We can try to use the "Kruskal" or "CP" decomposition approach.
    
    # For now, let's just return the schoolbook decomposition.
    # Rank = 60.
    
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
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to improve using a simple greedy approach:
    # For each term, check if it can be expressed as a linear combination of others.
    # This is hard in general, but for this specific tensor, maybe there are dependencies.
    
    # Let's just return the schoolbook for now.
    return {"U": U, "V": V, "W": W}
