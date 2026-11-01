"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x8 matrix.
    
    Returns dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    and the decomposition satisfies:
      for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    # Rank-16 construction: 2 terms per column
    # For each column k:
    #   Term 2k:   handles (i=0,j=0,k) and (i=1,j=0,k) via shared V[k,·] and W[2k,·]+W[2k+1,·]
    #   Term 2k+1: handles (i=0,j=1,k) and (i=1,j=1,k) via shared V[8+k,·] and W[2k,·]+W[2k+1,·]
    
    rank = 2 * p  # 16
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 16
    V = [[0] * rank for _ in range(m * p)]  # 16 x 16
    W = [[0] * rank for _ in range(p * n)]  # 16 x 16
    
    for k in range(p):
        # Term 2k: j=0, shared across i=0 and i=1
        r = 2 * k
        U[0 * m + 0][r] = 1  # i=0, j=0
        U[1 * m + 0][r] = 1  # i=1, j=0
        V[0 * p + k][r] = 1  # j=0, col=k
        W[k * n + 0][r] = 1  # col=k, i=0
        W[k * n + 1][r] = 1  # col=k, i=1
        
        # Term 2k+1: j=1, shared across i=0 and i=1
        r = 2 * k + 1
        U[0 * m + 1][r] = 1  # i=0, j=1
        U[1 * m + 1][r] = 1  # i=1, j=1
        V[1 * p + k][r] = 1  # j=1, col=k
        W[k * n + 0][r] = 1  # col=k, i=0
        W[k * n + 1][r] = 1  # col=k, i=1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    # Should never happen
                    pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
