"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 145  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-35 exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
    
    Uses Strassen's rank-7 algorithm on each of 5 independent 2x2 column blocks.
    Total rank = 5 * 7 = 35.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer entries.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    num_blocks = p // 2   # 5 blocks
    rank = 7 * num_blocks  # 35

    U = [[0] * rank for _ in range(n * m)]   # (4, 35)
    V = [[0] * rank for _ in range(m * p)]   # (20, 35)
    W = [[0] * rank for _ in range(p * n)]   # (20, 35)

    # Strassen's 7 products for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    #
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    #
    # U indices: i*m+j -> a=0, b=1, c=2, d=3
    # V indices: j*p+k -> e=V[0*10+2b], f=V[0*10+2b+1], g=V[1*10+2b], h=V[1*10+2b+1]
    # W indices: k*n+i -> C[0,2b]=W[4b], C[0,2b+1]=W[4b+2], C[1,2b]=W[4b+1], C[1,2b+1]=W[4b+3]

    for b in range(num_blocks):
        base = 7 * b

        # P1 = (a + d)(e + h)
        col = base
        U[0][col] = 1   # a
        U[3][col] = 1   # d
        V[2*b][col] = 1          # e
        V[10 + 2*b + 1][col] = 1 # h
        W[4*b][col] = 1          # C[0, 2b]
        W[4*b+3][col] = 1        # C[1, 2b+1]

        # P2 = (c + d)e
        col = base + 1
        U[2][col] = 1   # c
        U[3][col] = 1   # d
        V[2*b][col] = 1 # e
        W[4*b+1][col] = 1   # C[1, 2b]
        W[4*b+3][col] = -1  # C[1, 2b+1]

        # P3 = a(f - h)
        col = base + 2
        U[0][col] = 1   # a
        V[2*b+1][col] = 1         # f
        V[10 + 2*b + 1][col] = -1 # -h
        W[4*b+2][col] = 1         # C[0, 2b+1]
        W[4*b+3][col] = 1         # C[1, 2b+1]

        # P4 = d(g - e)
        col = base + 3
        U[3][col] = 1   # d
        V[10 + 2*b][col] = 1    # g
        V[2*b][col] = -1        # -e
        W[4*b][col] = 1         # C[0, 2b]
        W[4*b+1][col] = 1       # C[1, 2b]

        # P5 = (a + b)h
        col = base + 4
        U[0][col] = 1   # a
        U[1][col] = 1   # b
        V[10 + 2*b + 1][col] = 1 # h
        W[4*b][col] = -1         # C[0, 2b]
        W[4*b+2][col] = 1        # C[0, 2b+1]

        # P6 = (c - a)(e + f)
        col = base + 5
        U[2][col] = 1    # c
        U[0][col] = -1   # -a
        V[2*b][col] = 1   # e
        V[2*b+1][col] = 1 # f
        W[4*b+3][col] = 1 # C[1, 2b+1]

        # P7 = (b - d)(g + h)
        col = base + 6
        U[1][col] = 1    # b
        U[3][col] = -1   # -d
        V[10 + 2*b][col] = 1    # g
        V[10 + 2*b + 1][col] = 1 # h
        W[4*b][col] = 1         # C[0, 2b]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
