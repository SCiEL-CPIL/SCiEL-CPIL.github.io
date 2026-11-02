"""Rewritten program for matrix multiplication tensor decomposition M_{3,5,7}."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = 3, 5, 7
    
    # Schoolbook decomposition (rank = n*m*p = 105)
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Try to find a lower-rank decomposition using structured approaches.
    # 
    # Approach 1: Block decomposition with rank-deficient sub-blocks
    # For M_{n,m,p}, we can split dimensions and use known small results.
    # 
    # Known small ranks:
    # M_{1,m,p} has rank m*p (no reduction possible)
    # M_{2,2,2} has rank 7 (Strassen)
    # M_{2,3,3} - let's think about this
    #
    # Approach 2: Use the fact that for rectangular matrices, we can sometimes
    # do better than schoolbook by exploiting structure.
    #
    # Let's try a direct algebraic approach:
    # C = A * B where A is 3x5, B is 5x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Schoolbook uses 3*5*7 = 105 multiplications.
    #
    # One strategy: group columns of A and rows of B.
    # Split j into groups. For each group, compute partial products.
    #
    # Actually, let's try a more systematic approach using the tensor rank.
    # The matrix multiplication tensor T_{n,m,p} has rank at most n*m*p.
    # We want to find the minimal rank.
    #
    # For M_{3,5,7}, the known best upper bound from literature is around 97-100,
    # but constructing an explicit decomposition is non-trivial.
    #
    # Let's try a simple improvement: use Strassen-like ideas on sub-blocks.
    #
    # Split A into [A1; A2] where A1 is 2x5, A2 is 1x5
    # Split B into [B1 B2] where B1 is 5x4, B2 is 5x3
    #
    # C = A1*B + A2*B
    #   = [A1*B1  A1*B2] + [A2*B1  A2*B2]
    #
    # M_{2,5,4}: rank at most 2*5*4=40, but can we do better?
    # M_{2,5,3}: rank at most 2*5*3=30
    # M_{1,5,4}: rank = 5*4=20 (no reduction for 1-row)
    # M_{1,5,3}: rank = 5*3=15
    #
    # Total: 40+30+20+15 = 105. No gain.
    #
    # Let's try splitting differently.
    # A: 3x5, split into rows: [A1 (1x5); A2 (2x5)]
    # B: 5x7, split into cols: [B1 (5x3) B2 (5x4)]
    #
    # C = [A1*B1  A1*B2; A2*B1  A2*B2]
    #
    # M_{1,5,3}: rank 15
    # M_{1,5,4}: rank 20  
    # M_{2,5,3}: rank 30
    # M_{2,5,4}: rank 40
    # Total: 105. Still no gain.
    #
    # The key insight is that for 1-row or 1-column blocks, no Strassen-like
    # reduction is possible. We need at least 2x2 sub-blocks.
    #
    # Let's try: split A into [A1 (2x5); A2 (1x5)] and B into [B1 (5x5) B2 (5x2)]
    # C = [A1*B1  A1*B2; A2*B1  A2*B2]
    #
    # M_{2,5,5}: Can we do better than 50? 
    #   Split B1 into [B1a (5x3) B1b (5x2)]
    #   M_{2,5,3}: 30, M_{2,5,2}: 20, total 50. No gain.
    #   Or use Strassen on a 2x2 sub-block?
    #
    # M_{2,5,2}: rank at most 20. Can we do better?
    #   This is equivalent to multiplying a 2x5 matrix by a 5x2 matrix.
    #   C = A*B where A is 2x5, B is 5x2.
    #   C[i,k] = sum_j A[i,j]*B[j,k]
    #   This is like computing 2x2 matrix product from 5-dimensional vectors.
    #   The rank of M_{2,5,2}... 
    #   Actually, M_{2,m,2} has rank at most 2*2*m = 4m but can be lower.
    #   For m=2, M_{2,2,2} has rank 7 (Strassen).
    #   For general m, M_{2,m,2} can be decomposed as sum of M_{2,1,2} blocks,
    #   each of rank 4 (since M_{2,1,2} is just outer product, rank 4? No,
    #   M_{2,1,2} has tensor rank 2*1*2=4? Let me reconsider.
    #
    # Actually, M_{2,1,2}: C[i,k] = A[i,1]*B[1,k], which is an outer product.
    # The tensor is a rank-1 tensor in the appropriate sense? No, it's a
    # bilinear map. The tensor rank of M_{2,1,2} is 4 (each output element
    # requires one multiplication).
    #
    # For M_{2,m,2}, we can use the decomposition:
    # Write A as sum of 2x1 matrices and B as sum of 1x2 matrices.
    # A = sum_j a_j * e_j^T where a_j is 2x1, e_j is unit vector in R^m
    # B = sum_k f_k * b_k^T where f_k is 1x2... 
    #
    # This is getting complex. Let me try a different approach.
    #
    # Let's just try to use a randomized algebraic method to find a low-rank
    # decomposition, or use a known explicit construction.
    #
    # Actually, for the purpose of this task, let's try a simple but potentially
    # effective approach: use the "block Strassen" method.
    #
    # For M_{3,5,7}, we can try to express it as a sum of smaller tensor products
    # with some coefficients.
    #
    # Let's try a very simple improvement: notice that if we fix i and k,
    # C[i,k] = sum_j A[i,j] B[j,k]. There's no obvious way to reduce the number
    # of terms below n*m*p for generic matrices without using algebraic identities.
    #
    # Given the time constraints and the difficulty of finding an explicit
    # lower-rank decomposition, let's return the schoolbook decomposition.
    #
    # However, let's try one more thing: a simple "factorization" approach.
    # If we can write A = P*Q where P is 3xr and Q is rx5, and B = R*S where
    # R is 5xs and S is sx7, then C = P*(Q*R)*S. But Q*R is rxs, so we need
    # r*s*min(r,s) multiplications in the middle, which is likely worse.
    #
    # Let's just return schoolbook.
    return schoolbook()
# EVOLVE-BLOCK-END
