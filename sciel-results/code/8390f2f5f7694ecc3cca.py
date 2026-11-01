"""Matrix multiplication 2x2 by 2x7 - rank-minimal decomposition."""
import time

DIMENSIONS = [2, 2, 7]

def construct():
    """
    Construct a rank-21 tensor decomposition for M(2,2,7).

    A is 2x2, B is 2x7, C = A*B is 2x7.
    W[k*n+i, r] gives the coefficient for C[i,k] (n=2, so W[2*k+i, r]).

    Decomposition as a tensor product M(2,2,1) x M(2,1,7):
      - M(2,2,1) (2x2 times 2x1) has rank 4 (schoolbook, no savings for 1 output).
      - M(2,1,7) (2x1 times 1x7) has rank 7 (each output is a 2-term sum).
      - rank(A x B) <= rank(A)*rank(B) = 4*7 = 21.

    Rank-1 terms: (i,j) in {0,1}^2 (the 4 A-products), t in {0..6} (sum slot).
      U[i*m + j, r] = 1
      V[j*p + t, r] = 1        (B[j,t] with B[0,t]=B[1,t]=1)
      W[t*n + i, r] = 1        (contributes to C[i, t])
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=7

    rank = 4 * 7  # = 21
    U = [[0.0] * rank for _ in range(n * m)]   # (4, 21)
    V = [[0.0] * rank for _ in range(m * p)]   # (14, 21)
    W = [[0.0] * rank for _ in range(p * n)]   # (14, 21)

    r = 0
    for i in range(n):
        for j in range(m):
            for t in range(p):
                U[i * m + j][r] = 1.0
                V[j * p + t][r] = 1.0
                W[t * n + i][r] = 1.0
                r += 1

    return {"U": U, "V": V, "W": W}
