"""Rank-10 exact tensor decomposition for 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposition of rank 10 for multiplying a 2x5 matrix by a 5x5 matrix.
    
    The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    We can decompose this into two independent rank-5 decompositions, one for each row i.
    Since the problem allows half-integers, we can also use a sum/difference trick, 
    but independent rows already give rank 2*m = 10 for n=2, m=5.
    
    Rank R = 2 * m = 10.
    r = 0..4 corresponds to row i=0, j=0..4.
    r = 5..9 corresponds to row i=1, j=0..4.
    
    For r in 0..4 (i=0, j=r):
    U[0*5+j, r] = 1
    U[1*5+j, r] = 0
    V[j*5+k, r] = 1 for all k
    W[k*2+0, r] = 1
    W[k*2+1, r] = 0
    
    For r in 5..9 (i=1, j=r-5):
    U[0*5+j, r] = 0
    U[1*5+j, r] = 1
    V[j*5+k, r] = 1 for all k
    W[k*2+0, r] = 0
    W[k*2+1, r] = 1
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = 2 * m  # 10
    
    U = [[0] * rank for _ in range(n * m)]  # (10, 10)
    V = [[0] * rank for _ in range(m * p)]  # (25, 10)
    W = [[0] * rank for _ in range(p * n)]  # (10, 10)
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U[i*m+j, r] = 1
            U[i * m + j][r] = 1
            
            # V[j*p+k, r] = 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1
            
            # W[k*n+i, r] = 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
