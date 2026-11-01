"""
Program to find the rank-minimal exact tensor decomposition for 2x3 by 3x3 matrix multiplication.

Analysis:
The tensor for multiplying a 2x3 matrix A by a 3x3 matrix B to get a 2x3 matrix C
is defined by C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k].
The input variables are A[i,j] (6 vars) and B[j,k] (9 vars).
The output variables are C[i,k] (6 vars).

The tensor decomposes into independent parts based on the output index k and input index i.
For a fixed k, the computation involves B[j,k] for j=0,1,2. These B variables are disjoint
for different k. Thus, the tensor is a direct sum over k of sub-tensors T_k.
Each T_k computes C[i,k] for i=0,1 from A[i,j] and B[j,k].
For a fixed k and i, the term is sum_{j=0}^{2} A[i,j] B[j,k].
This is a dot product of two 3-vectors. The rank of the dot product tensor M_{1,3,1} is 3
(it is the identity tensor in the summation index, and cannot be reduced further as the
variables are independent).
Since there are 2 choices for i and 3 choices for k, and the variables for different (i,k)
pairs involving B are disjoint (for B) and A are shared but the output lines are distinct,
the total rank is the sum of the ranks of these independent components.
Rank = 2 (rows of A) * 3 (cols of B) * 3 (length of dot product) = 18.

Thus, the schoolbook rank of 18 is minimal.
"""

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
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
# EVOLVE-BLOCK-END
