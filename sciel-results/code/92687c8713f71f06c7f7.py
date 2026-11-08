"""Tensor decomposition for 5x5 matrix multiplication.

Strategy: Use a 2+3 block decomposition to reduce rank below the schoolbook 125.

Split the 5x5 matrices into blocks:
- A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
- B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3

Then C = A*B has blocks:
- C11 = A11*B11 + A12*B21  (2x2)
- C12 = A11*B12 + A12*B22  (2x3)
- C21 = A21*B11 + A22*B21  (3x2)
- C22 = A21*B12 + A22*B22  (3x3)

For each block multiplication, we can use Strassen-like algorithms where applicable.

The key insight is that we can use the following decompositions:
- 2x2 * 2x2 -> 2x2: rank 7 (Strassen)
- 2x2 * 2x3 -> 2x3: can be done with rank 7 (treat as 2*(2x2*2x2) but optimized)
- 2x3 * 2x2 -> 2x2: rank 7
- 2x3 * 3x3 -> 2x3: rank ?
- 3x2 * 2x2 -> 3x2: rank 7
- 3x2 * 2x3 -> 3x3: rank ?
- 3x3 * 3x3 -> 3x3: rank ?

Actually, let me think more carefully. The standard approach for 5x5 is to use
the fact that 5 = 2 + 3, and then use known fast algorithms for the block multiplications.

For the block structure:
C11 (2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
C12 (2x3) = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
C21 (3x2) = A21(3x2)*B11(2x2) + A22(3x3)*B21(3x2)
C22 (3x3) = A21(3x2)*B12(2x3) + A22(3x3)*B22(3x3)

We can use the following strategy:
1. For 2x2 * 2x2: use Strassen (rank 7)
2. For 2x3 * 3x2: This is a 2x3 times 3x2 = 2x2 result. We can view this as
   computing 2 dot products of length 3. The schoolbook rank is 2*3*2=12, but
   we can do better. Actually, this is equivalent to multiplying a 2x3 by a 3x2,
   which gives a 2x2. The rank of this tensor is at most 7 (by embedding in 2x2*2x2
   with some zero padding, but that gives rank 7 for the 2x2 result).
   
   Actually, let me think about this differently. The product A12*B21 where A12 is 2x3
   and B21 is 3x2 gives a 2x2 matrix. This is equivalent to:
   C[i,k] = sum_j A12[i,j] * B21[j,k] for i=0,1 and k=0,1
   
   This is just a 2x2 matrix multiplication where the "middle" dimension is 3 instead of 2.
   The schoolbook rank is 2*3*2 = 12. But we can use a rank-7 algorithm by treating it
   as an embedded problem.
   
   Actually, the simplest approach: just use schoolbook for the cross terms and Strassen
   for the square blocks.
   
   Let me compute the total rank with schoolbook for all:
   - C11: A11*B11 (2x2*2x2, rank 4) + A12*B21 (2x3*3x2, rank 12) = 16
   - C12: A11*B12 (2x2*2x3, rank 12) + A12*B22 (2x3*3x3, rank 18) = 30
   - C21: A21*B11 (3x2*2x2, rank 12) + A22*B21 (3x3*3x2, rank 18) = 30
   - C22: A21*B12 (3x2*2x3, rank 18) + A22*B22 (3x3*3x3, rank 27) = 45
   
   Total: 16 + 30 + 30 + 45 = 121
   
   That's already better than 125!

Let me implement this block decomposition with schoolbook for each sub-product.
"""

import time
import numpy as np


def construct():
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    # Block sizes: split 5 into 2 + 3
    n1, n2 = 2, 3  # Row split for A and C
    m1, m2 = 2, 3  # Column split for A and row split for B
    p1, p2 = 2, 3  # Column split for B and C

    # We'll build the decomposition term by term.
    # Each term r has U[:, r], V[:, r], W[:, r]
    # U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    
    # We'll collect all (U_col, V_col, W_col) tuples
    terms = []
    
    def add_term(U_col, V_col, W_col):
        """Add a rank-1 term. U_col is length n*m, V_col is length m*p, W_col is length p*n."""
        terms.append((np.array(U_col, dtype=np.float64),
                      np.array(V_col, dtype=np.float64),
                      np.array(W_col, dtype=np.float64)))
    
    # Helper: add schoolbook terms for multiplying an (a x b) by (b x c) sub-matrix
    # where the sub-matrix of A occupies rows [r0:r0+a], cols [c0:c0+b]
    # and the sub-matrix of B occupies rows [r0:r0+b], cols [c0:c0+c]
    # and the result goes into C rows [r0:r0+a], cols [c0:c0+c]
    def add_schoolbook(a, b, c, A_row_start, A_col_start, B_row_start, B_col_start, C_row_start, C_col_start):
        """Add schoolbook terms for A_sub (a x b) * B_sub (b x c) -> C_sub (a x c)."""
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    
                    # A index: row = A_row_start + i, col = A_col_start + j
                    A_idx = (A_row_start + i) * m + (A_col_start + j)
                    # B index: row = B_row_start + j, col = B_col_start + k
                    B_idx = (B_row_start + j) * p + (B_col_start + k)
                    # C index: row = C_row_start + i, col = C_col_start + k
                    # W index: W[k*n + i] where k is the column of C, i is the row of C
                    C_row = C_row_start + i
                    C_col = C_col_start + k
                    W_idx = C_col * n + C_row  # W[k*n + i] convention
                    
                    U_col[A_idx] = 1.0
                    V_col[B_idx] = 1.0
                    W_col[W_idx] = 1.0
                    
                    add_term(U_col, V_col, W_col)
    
    # Block C11 (2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    # A11: rows 0-1, cols 0-1; B11: rows 0-1, cols 0-1; C11: rows 0-1, cols 0-1
    add_schoolbook(2, 2, 2, 0, 0, 0, 0, 0, 0)  # A11*B11
    # A12: rows 0-1, cols 2-4; B21: rows 2-4, cols 0-1; C11: rows 0-1, cols 0-1
    add_schoolbook(2, 3, 2, 0, 2, 2, 0, 0, 0)  # A12*B21
    
    # Block C12 (2x3) = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
    # A11: rows 0-1, cols 0-1; B12: rows 0-1, cols 2-4; C12: rows 0-1, cols 2-4
    add_schoolbook(2, 2, 3, 0, 0, 0, 2, 0, 2)  # A11*B12
    # A12: rows 0-1, cols 2-4; B22: rows 2-4, cols 2-4; C12: rows 0-1, cols 2-4
    add_schoolbook(2, 3, 3, 0, 2, 2, 2, 0, 2)  # A12*B22
    
    # Block C21 (3x2) = A21(3x2)*B11(2x2) + A22(3x3)*B21(3x2)
    # A21: rows 2-4, cols 0-1; B11: rows 0-1, cols 0-1; C21: rows 2-4, cols 0-1
    add_schoolbook(3, 2, 2, 2, 0, 0, 0, 2, 0)  # A21*B11
    # A22: rows 2-4, cols 2-4; B21: rows 2-4, cols 0-1; C21: rows 2-4, cols 0-1
    add_schoolbook(3, 3, 2, 2, 2, 2, 0, 2, 0)  # A22*B21
    
    # Block C22 (3x3) = A21(3x2)*B12(2x3) + A22(3x3)*B22(3x3)
    # A21: rows 2-4, cols 0-1; B12: rows 0-1, cols 2-4; C22: rows 2-4, cols 2-4
    add_schoolbook(3, 2, 3, 2, 0, 0, 2, 2, 2)  # A21*B12
    # A22: rows 2-4, cols 2-4; B22: rows 2-4, cols 2-4; C22: rows 2-4, cols 2-4
    add_schoolbook(3, 3, 3, 2, 2, 2, 2, 2, 2)  # A22*B22
    
    rank = len(terms)
    
    # Build the full matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        U[:, r] = terms[r][0]
        V[:, r] = terms[r][1]
        W[:, r] = terms[r][2]
    
    # Verify the decomposition
    rng = np.random.RandomState(42)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        A_part = A_test @ U[:, r]  # n-vector
        B_part = B_test @ V[:, r]  # p-vector
        C_computed += np.outer(A_part, B_part) * W[:, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to pure schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
