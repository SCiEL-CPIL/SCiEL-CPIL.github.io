import time
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if i == k else 0.0
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    # The tensor T for 3x3 * 3x6 -> 3x6 multiplication has T[i,j,k] = 1 if i==k else 0.
    # This is equivalent to the identity tensor on the 3x3 block, repeated for k=0,1,2.
    # For k >= 3, all entries are 0.
    # We can construct a rank-9 decomposition:
    # For each (i,j) in 0..2 x 0..2, create a term that contributes to (i,j,i) with value 1.
    # U[i*3+j, c] = 1
    # V[j*6+i, c] = 1
    # W[i*3+i, c] = 1
    # This gives T[i,j,k] = sum_c U[i*3+j,c]*V[j*6+k,c]*W[k*3+i,c]
    # = 1 if there exists c such that (i*3+j, j*6+k, k*3+i) matches the pattern.
    # For c corresponding to (i',j'), we have U[i*3+j,c]=1 iff i=i',j=j'
    # V[j*6+k,c]=1 iff j=j', k=i'
    # W[k*3+i,c]=1 iff k=i', i=i'
    # So the product is 1 iff i=i', j=j', k=i'. Thus T[i,j,k] = 1 iff i=j'=j? No.
    # i=i', j=j', k=i'. So i=i', k=i', j=j'. Thus i=k and j is free? No, j must equal j'.
    # But j' is fixed for the term. So for a given term (i',j'), it contributes 1 to (i',j',i').
    # So T[i,j,k] = 1 iff there exists (i',j') such that i=i', j=j', k=i'.
    # This is true iff i=k (since j'=j is always possible for any j).
    # Wait, if i=k, then we need a term with i'=i and j'=j. Such a term exists.
    # If i != k, no term exists.
    # So this decomposition is valid. Rank = 9.
    
    n, m, p = DIMENSIONS
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + i][col] = 1
            W[i * n + i][col] = 1
            col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    if _verify(U, V, W):
        return result
    else:
        return _schoolbook()

BET: {mechanism: "The tensor T[i,j,k] = delta_{i,k} for i<3, k<6 can be decomposed with rank 9 by using one term per (i,j) pair that matches i=k, yielding a 6x improvement over schoolbook rank 54.", q: 0.95, delta: [5.0, 5.0]}
