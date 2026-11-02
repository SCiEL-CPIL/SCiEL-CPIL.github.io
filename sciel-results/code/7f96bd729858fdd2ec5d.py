"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 4
    U = [[0] * rank for _ in range(n * m)]  # 4 x 4
    V = [[0] * rank for _ in range(m * p)]  # 12 x 4
    W = [[0] * rank for _ in range(p * n)]  # 12 x 4
    
    # c=0: A[0,0]*B[0,k] for all k
    U[0*2+0][0] = 1  # i=0, j=0
    for k in range(p):
        V[0*p+k][0] = 1  # j=0, k
        W[k*n+0][0] = 1  # k, i=0
    
    # c=1: A[0,1]*B[1,k] for all k
    U[0*2+1][1] = 1  # i=0, j=1
    for k in range(p):
        V[1*p+k][1] = 1  # j=1, k
        W[k*n+0][1] = 1  # k, i=0
    
    # c=2: A[1,0]*B[0,k] for all k
    U[1*2+0][2] = 1  # i=1, j=0
    for k in range(p):
        V[0*p+k][2] = 1  # j=0, k
        W[k*n+1][2] = 1  # k, i=1
    
    # c=3: A[1,1]*B[1,k] for all k
    U[1*2+1][3] = 1  # i=1, j=1
    for k in range(p):
        V[1*p+k][3] = 1  # j=1, k
        W[k*n+1][3] = 1  # k, i=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
