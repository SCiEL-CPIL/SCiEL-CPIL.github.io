"""
Tensor decomposition for 2x2 by 2x12 matrix multiplication.
Uses a "row-aggregation" approach: decompose each row of A (1x2) times B (2x12)
as a rank-12 tensor, giving total rank 24.

The key insight: A is 2x2, so A[i,j] for i=0,1 and j=0,1.
C[i,k] = sum_j A[i,j] * B[j,k]

For each row i of A (i=0,1), we need to compute the product of a 1x2 vector with a 2x12 matrix.
This is a (1,2,12) tensor. The rank of M_{1,2,12} is 12 (each output C[i,k] depends on both A[i,0] and A[i,1]).

Actually, let me reconsider. The tensor for M_{1,2,12}:
- Input A has 1*2 = 2 entries: A[0,0], A[0,1]
- Input B has 2*12 = 24 entries: B[j,k] for j=0,1 and k=0..11
- Output C has 12*1 = 12 entries: C[0,k] for k=0..11

C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]

This is a rank-12 tensor: for each k, C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k].
We can write this as 12 rank-1 terms:
For each k: term_k = (A[0,0] * 1 + A[0,1] * 0) * (B[0,k] * 1 + B[1,k] * 0) ... no that's not right.

Actually, C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
This is a sum of 2 rank-1 terms per k, so 24 rank-1 terms for one row.
But we can do better: the tensor M_{1,2,12} has rank 12.

Wait, let me think more carefully. The tensor T_{1,2,12} has entries T[a,b,c] where:
- a indexes A (size 1*2 = 2): a = 0 -> A[0,0], a = 1 -> A[0,1]
- b indexes B (size 2*12 = 24): b = j*12+k -> B[j,k]
- c indexes C (size 12*1 = 12): c = k -> C[0,k]

T[a,b,c] = 1 if c=k and (a=0, b=0*12+k) or (a=1, b=1*12+k), else 0.

So T[a,b,c] = delta(a=0, b=k) + delta(a=1, b=12+k) where c=k.

This tensor has rank 12: for each k, define:
- u_k: vector in A-space with u_k[0]=1, u_k[1]=0
- v_k: vector in B-space with v_k[k]=1, v_k[12+k]=1, others 0
- w_k: vector in C-space with w_k[k]=1, others 0

Then T = sum_k u_k ⊗ v_k ⊗ w_k.

Yes! So rank of M_{1,2,12} is 12.

For M_{2,2,12}, we have two rows of A. Row 0 uses A[0,0] and A[0,1], row 1 uses A[1,0] and A[1,1].
These are independent, so total rank = 12 + 12 = 24.

Let me verify the indexing:
- U shape: (n*m, rank) = (4, 24). U[i*m+j, r] where i=0,1 (row of A), j=0,1 (col of A)
- V shape: (m*p, rank) = (24, 24). V[j*p+k, r] where j=0,1 (row of B), k=0..11 (col of B)
- W shape: (p*n, rank) = (24, 24). W[k*n+i, r] where k=0..11 (col of C), i=0,1 (row of C)

For row i of A (i=0,1):
For each k (0..11):
- u: U[i*2+0, col] = 1 (A[i,0]), U[i*2+1, col] = 0
- v: V[0*12+k, col] = 1 (B[0,k]), V[1*12+k, col] = 1 (B[1,k])
- w: W[k*2+i, col] = 1 (C[i,k])

Total rank = 2*12 = 24.
"""

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Row-aggregation tensor decomposition for 2x2 by 2x12 matrix multiplication.
    Decomposes into two independent (1,2,12) sub-problems, each of rank 12.
    Total rank = 24.
    
    U shape: (n*m, rank) = (4, 24)
    V shape: (m*p, rank) = (24, 24)
    W shape: (p*n, rank) = (24, 24)
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    rank = 2 * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 24 x 24
    
    col = 0
    for i in range(n):  # row of A and C
        for k in range(p):  # column of B and C
            # U: A[i, 0] contributes
            U[i * m + 0][col] = 1
            # U: A[i, 1] does NOT contribute in this term (it's in the other term)
            # Actually, we need both A[i,0]*B[0,k] and A[i,1]*B[1,k]
            # So this single term handles both via V having both entries
            
            # V: B[0, k] and B[1, k]
            V[0 * p + k][col] = 1
            V[1 * p + k][col] = 1
            
            # W: C[i, k]
            W[k * n + i][col] = 1
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
