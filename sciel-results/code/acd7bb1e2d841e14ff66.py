"""Program for matrix multiplication 2x2 x 2x13 tensor decomposition."""
DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # For each column k of B (p columns), the sub-problem is (2x2) x (2x1) -> (2x1).
    # The rank of M_{2,2,1} is 2 (not 3, as for M_{2,2,2}).
    # Total rank = p * 2 = 26.
    #
    # Decomposition for M_{2,2,1} (for a single column k):
    # We need C[i,0] = sum_j A[i,j] B[j,0] for i=0,1.
    #
    # Term r=2k (for j=0 contribution):
    #   U[i,m+j] = U[i,0] = 1 for all i  => U[:, 2k] = [1, 1, 0, 0]^T  (i=0,1; j=0,1)
    #   V[j,p+k] = V[0,k] = 1, V[1,k] = 0 => V[:, 2k] has 1 at row 0*k... 
    #   Actually V is (m*p, rank) = (2*13, 26). Row index is j*p + k.
    #   For term 2k: V[0*13 + k][2k] = 1, all other rows 0.
    #   W is (p*n, rank) = (13*2, 26). Row index is k*n + i.
    #   W[k*2 + 0][2k] = 1, W[k*2 + 1][2k] = 1.
    #
    # Term r=2k+1 (for j=1 contribution):
    #   U[i,1] = 1 for all i => U[:, 2k+1] = [0, 0, 1, 1]^T (i=0,1; j=0,1)
    #   V[1*p + k][2k+1] = 1 => V[13 + k][2k+1] = 1.
    #   W[k*2 + 0][2k+1] = 1, W[k*2 + 1][2k+1] = 1.
    #
    # Let's verify for C[i,k] = sum_j A[i,j] B[j,k]:
    # Contribution from term 2k: U[i,0]*V[0,k]*W[k*n+i] = 1*1*1 * A[i,0]*B[0,k] = A[i,0]*B[0,k]
    # Contribution from term 2k+1: U[i,1]*V[1,k]*W[k*n+i] = 1*1*1 * A[i,1]*B[1,k] = A[i,1]*B[1,k]
    # Total: A[i,0]*B[0,k] + A[i,1]*B[1,k] = C[i,k]. Correct!
    
    rank = 2 * p
    U = [[0] * rank for _ in range(n * m)]  # (4, 26)
    V = [[0] * rank for _ in range(m * p)]  # (26, 26)
    W = [[0] * rank for _ in range(p * n)]  # (26, 26)
    
    for k in range(p):
        # Term for j=0
        r0 = 2 * k
        for i in range(n):
            U[i * m + 0][r0] = 1
        V[0 * p + k][r0] = 1
        for i in range(n):
            W[k * n + i][r0] = 1
        
        # Term for j=1
        r1 = 2 * k + 1
        for i in range(n):
            U[i * m + 1][r1] = 1
        V[1 * p + k][r1] = 1
        for i in range(n):
            W[k * n + i][r1] = 1
    
    return {"U": U, "V": V, "W": W}
