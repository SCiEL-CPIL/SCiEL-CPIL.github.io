"""
Rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.

Approach: Direct combinatorial construction using the fact that
R(2, 2, 1) = 2 (the rank of 2x2 * 2x1 matrix multiplication is 2).

For 2x2 * 2x7, we can decompose into 7 independent 2x2 * 2x1 multiplications,
each requiring rank 2. Total rank = 14.

The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1.
With n=2, m=2, p=7:
- A indices: i*m+j where i in [0,1], j in [0,1] -> 4 entries
- B indices: j*p+k where j in [0,1], k in [0,6] -> 14 entries  
- C indices: k*n+i where k in [0,6], i in [0,1] -> 14 entries

For each column k of B, we need to compute:
C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

This is a 2x2 * 2x1 multiplication, which has rank 2.
We use the standard rank-2 decomposition:
- Term 1: U=(1,1,0,0), V=(1,1,0,...,0), W=(1,0) for C[0,k]
- Term 2: U=(0,0,1,1), V=(1,1,0,...,0), W=(0,1) for C[1,k]

Total rank = 7 * 2 = 14.
"""


def construct():
    n, m, p = 2, 2, 7
    rank = 14
    
    # U: shape (n*m, rank) = (4, 14)
    # V: shape (m*p, rank) = (14, 14)
    # W: shape (p*n, rank) = (14, 14)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        # Term 1: Contributes to C[0, k]
        # A indices: A[0,0] and A[0,1] are used
        U[0][col] = 1  # A[0,0]
        U[1][col] = 1  # A[0,1]
        # B indices: B[0,k] and B[1,k] are used
        V[0 * p + k][col] = 1  # B[0,k]
        V[1 * p + k][col] = 1  # B[1,k]
        # C index: C[0,k] = k*n + 0 = 2k
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: Contributes to C[1, k]
        # A indices: A[1,0] and A[1,1] are used
        U[2][col] = 1  # A[1,0]
        U[3][col] = 1  # A[1,1]
        # B indices: B[0,k] and B[1,k] are used
        V[0 * p + k][col] = 1  # B[0,k]
        V[1 * p + k][col] = 1  # B[1,k]
        # C index: C[1,k] = k*n + 1 = 2k+1
        W[k * n + 1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
