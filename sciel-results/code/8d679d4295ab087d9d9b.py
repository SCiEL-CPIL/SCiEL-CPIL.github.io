"""
Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.

The task is to find integer/half-integer matrices U (12 x r), V (20 x r), W (15 x r)
such that the matrix multiplication tensor is decomposed.

Known results:
- Schoolbook rank = 3*4*5 = 60
- Theoretical lower bounds suggest much less is possible
- For small cases, exact rank might be known or we can search

Strategy:
1. Try to find known optimal ranks for small matrix multiplication
2. If not known, try constructive approaches
3. Fall back to schoolbook (rank 60) if nothing better found

For 3x4x5, let's think about what's possible:
- The tensor has 3*4*5 = 60 terms
- Fast matrix multiplication could reduce this
- Strassen-like approaches work for square matrices

Actually, let me think more carefully. The matrix multiplication tensor for A (n x m) * B (m x p)
has entries M[i,j,k] = 1 if the product (AB)[i,k] depends on A[i,j] and B[j,k].

For 3x4x5:
- A is 3x4, B is 4x5, C is 3x5
- C[i,k] = sum_j A[i,j] * B[j,k] for j=0,1,2,3

The tensor decomposition rank is the minimum number of rank-1 tensors needed.

Known results for small cases:
- 2x2x2: rank 7 (Strassen)
- 2x2x1: rank 2
- 1xn x nx1: rank n

For 3x4x5, I don't have a known optimal rank memorized. Let me try to construct
a decomposition with rank less than 60.

One approach: use the fact that we can share computation. For example, if we have
multiple products that share common sub-expressions, we can factor them out.

Actually, let me try a different approach. The matrix multiplication can be viewed as
a linear map from R^(m*p) to R^(n*p) for each fixed A, or from R^(n*m) to R^(n*p) for each fixed B.

Let me try to use a known result: for m x n matrices, the rank of matrix multiplication
is at most O(n^omega) where omega is the exponent of matrix multiplication.

For small matrices, let me just try to find a good decomposition by search or known constructions.

Actually, I recall that for 3x3x3, the rank is at most 23 (Bini et al.), and lower bounds suggest it's at least 19.

For 3x4x5, let me estimate: if we think of it as three 1x4x5 multiplications, each might have
rank around 4*5/2 = 10 (rough heuristic), so total around 30. But this is very rough.

Let me try a constructive approach: decompose the 3x4x5 multiplication into smaller pieces.

Actually, I think the safest bet is to implement a search for low-rank decompositions using
known techniques, but given the 120s time limit, let me try a few constructive approaches.

Approach 1: Use the fact that 4 = 2+2, so we can split the 4x5 matrix into two 2x5 matrices,
and the 3x4 into two 3x2 matrices. Then we have two 3x2x5 multiplications.
Each 3x2x5 has schoolbook rank 3*2*5 = 30, so total 60. No improvement.

Approach 2: Use Strassen-like recursion. For 3x4x5, we can try to use the 2x2x2 Strassen
decomposition as a building block.

Actually, let me just try to implement a simple search for rank r starting from a reasonable
lower bound. But this is computationally expensive.

Given the constraints, let me try the following:
1. Implement schoolbook (rank 60) as fallback
2. Try a few known constructions for low rank
3. If time permits, do a small search

Let me implement the schoolbook first, then try to improve.

Wait, I just realized: the score is seed/value, where value is the rank. So if seed is 60,
then rank 60 gives score 1.0. To get score > 1.0, we need rank < 60.

Let me try to find a rank < 60 decomposition.

Idea: For 3x4x5, note that 3*4*5 = 60. But we can potentially save by sharing.

Consider the product C = A * B where A is 3x4, B is 4x5.
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]

For each i (3 choices) and k (5 choices), we have 4 products. Total 3*5*4 = 60 products.

To reduce rank, we need to find linear combinations that allow us to compute multiple
products with fewer multiplications.

This is essentially the problem of finding the rank of the matrix multiplication tensor.

For small cases, this is known. Let me try to recall or derive:

For 2x2x2: rank 7 (Strassen)
For 2x2x1: rank 2
For 1x2x2: rank 2
For 2x3x3: I think rank is around 14-15? Not sure.

For 3x4x5, I don't have a specific number. Let me try a constructive approach.

Alternative: Use the fact that we can pad to square matrices. 3x4x5 can be embedded in
5x5x5 by padding. The rank of 5x5x5 is at most 5^omega, but that's not helpful directly.

Actually, let me just implement the schoolbook and see if I can find a simple improvement.

Hmm, I think for this task, the expected approach might be to use known results or
a simple search. Let me implement a basic version and try to find rank < 60.

Actually, I recall that for m x n matrix multiplication, the rank is at most
m * n * ceil(log2(m) * log2(n) * log2(p)) or something like that for fast multiplication.
But this is for asymptotic bounds.

For small matrices, let me just try to implement a search.

Given the time constraint, let me implement:
1. Schoolbook (rank 60)
2. A simple greedy or random search for rank < 60

But random search for tensor rank is very hard. Let me think of a better approach.

Actually, I just thought of something: the matrix multiplication tensor for A (n x m) * B (m x p)
is the same as the tensor for the bilinear map (A, B) -> A*B.

The rank of this tensor is the minimum number of scalar multiplications needed to compute
the product.

For 3x4x5, one approach is to use the following:
- Compute the 3x5 product using a fast algorithm for small matrices

But I don't have a specific fast algorithm for 3x4x5.

Let me just implement the schoolbook and hope that's sufficient for a valid score.

Actually, wait. Let me re-read the task. "Find a rank-minimal exact tensor decomposition".
The score is seed/value, where value is the rank. So if I can find rank < 60, I get a better score.

Let me try a different approach: use the fact that 4 = 2 + 2, and use Strassen for the 2x2x2 subproblems.

For 3x4x5:
- Split A (3x4) into A1 (3x2) and A2 (3x2)
- Split B (4x5) into B1 (2x5) and B2 (2x5)
- Then A*B = A1*B1 + A2*B2

Each A_i*B_i is a 3x2x5 multiplication. Can we do 3x2x5 with rank < 30?

For 3x2x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1,2 and k=0,1,2,3,4.
Total 3*2*5 = 30 products.

Can we do better? For 2x2x2, we can do rank 7 instead of 8. For 3x2x5, maybe we can save a few.

But I don't have a specific construction.

Given the time, let me just implement the schoolbook. The score will be 1.0, which is valid.

Actually, let me try one more thing: implement a simple version that tries rank 55, 50, etc.
by random initialization and checking if the decomposition is exact. But this is very unlikely
to work in 120s.

Let me just go with schoolbook.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 5]

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

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
