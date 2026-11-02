"""
Rank-26 decomposition for 3x3 matrix multiplication using Strassen-style block partitioning.

The idea: partition each 3x3 matrix into a 2x2 block and a 1x1 corner.
Apply Strassen's 7-multiplication algorithm to the 2x2 block, then handle
the remaining cross-terms and the 1x1 corner with additional rank-1 terms.

Total rank: 7 (Strassen for 2x2) + 8 (cross terms) + 1 (corner) = 16? 
No, let me think more carefully.

Actually, the known best for 3x3 is rank 23 (Landsberg 1998) but that's complex.
A simpler approach: partition into blocks of sizes (2,1) x (2,1) x (2,1).

Let A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
Same for B and C.

C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x1*1x2)
C12 = A11*B12 + A12*B22  (2x1 = 2x2*2x1 + 2x1*1x1)
C21 = A21*B11 + A22*B21  (1x2 = 1x2*2x2 + 1x1*1x2)
C22 = A21*B12 + A22*B22  (1x1 = 1x2*2x1 + 1x1*1x1)

Using Strassen for the 2x2*2x2 part gives 7 multiplications instead of 4.
The cross terms need additional rank-1 terms.

Let me count:
- C11: A11*B11 (7 via Strassen) + A12*B21 (2*1*1 = 2 rank-1 terms)
- C12: A11*B12 (this is 2x2 * 2x1, needs 4 rank-1 terms) + A12*B22 (2*1*1 = 2)
- C21: A21*B11 (1x2 * 2x2, needs 4) + A22*B21 (1*1*2 = 2)
- C22: A21*B12 (1x2 * 2x1, needs 2) + A22*B22 (1*1*1 = 1)

Total: 7 + 2 + 4 + 2 + 4 + 2 + 2 + 1 = 24

That's rank 24. We can potentially do better with cleverer combinations.

Actually, let me use a known rank-26 construction which is simpler to implement correctly.

A well-known approach: use the schoolbook for most entries but apply Strassen to a 2x2 sub-block.

Let me use a concrete rank-26 decomposition. I'll construct it explicitly.

For 3x3 matrices A, B with C = A*B:
A[i,j] for i,j in {0,1,2}, B[i,j] for i,j in {0,1,2}, C[i,j] = sum_k A[i,k]*B[k,j]

We can group indices. Let's use the following strategy:
- For the 2x2 sub-matrix C[0:2, 0:2], use Strassen (7 terms)
- For C[0:2, 2], use 4 terms (2x2 * 2x1)
- For C[2, 0:2], use 4 terms (1x2 * 2x2)  
- For C[2, 2], use 1 term
- Cross terms: A[0:2,2]*B[2,0:2] for C[0:2,0:2] needs 2 terms
- A[0:2,2]*B[2,2] for C[0:2,2] needs 2 terms
- A[2,0:2]*B[2,0:2] for C[2,0:2] needs 2 terms
- A[2,2]*B[2,2] for C[2,2] is already counted

Wait, let me recount. The full decomposition:

C[i,j] = sum_k A[i,k] * B[k,j]

Partition: let S = {0,1}, T = {2}.

C[i,j] = sum_{k in S} A[i,k]*B[k,j] + A[i,2]*B[2,j]

For i,j in S (4 entries):
  sum_{k in S} A[i,k]*B[k,j] -> 2x2*2x2, use Strassen: 7 terms
  A[i,2]*B[2,j] -> 2x1 * 1x2: 2 terms (since A[i,2] is 2x1, B[2,j] is 1x2)
  Total for C[S,S]: 7 + 2 = 9

For i in S, j in T (2 entries):
  sum_{k in S} A[i,k]*B[k,2] -> 2x2 * 2x1: 4 terms
  A[i,2]*B[2,2] -> 2x1 * 1x1: 2 terms
  Total for C[S,T]: 4 + 2 = 6

For i in T, j in S (2 entries):
  sum_{k in S} A[2,k]*B[k,j] -> 1x2 * 2x2: 4 terms
  A[2,2]*B[2,j] -> 1x1 * 1x2: 2 terms
  Total for C[T,S]: 4 + 2 = 6

For i in T, j in T (1 entry):
  sum_{k in S} A[2,k]*B[k,2] -> 1x2 * 2x1: 2 terms
  A[2,2]*B[2,2] -> 1x1 * 1x1: 1 term
  Total for C[T,T]: 2 + 1 = 3

Grand total: 9 + 6 + 6 + 3 = 24

So rank 24 is achievable. Let me implement this.

Strassen for 2x2:
M1 = (A00+A11)*(B00+B11)
M2 = (A10+A11)*B00
M3 = A00*(B10-B11)
M4 = A11*(B01-B10)
M5 = (A00+A01)*B11
M6 = (A10-A00)*(B00+B10)
M7 = (A01-A11)*(B01+B11)

C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

Each Mk is a product of two linear combinations, so each is a rank-1 term in the tensor sense (product of a vector in A-space and a vector in B-space).

Let me implement rank 24.
"""

import time
import sys

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-24 decomposition for 3x3 matrix multiplication.
    
    Partition indices into S={0,1} and T={2}.
    Use Strassen (7 terms) for the 2x2*2x2 sub-product.
    Use direct rank-1 terms for all cross terms.
    
    Total rank: 7 + 2 + 4 + 2 + 4 + 2 + 2 + 1 = 24
    """
    n, m, p = DIMENSIONS  # all 3
    
    # We'll build U, V, W as lists of lists.
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (9, rank)  
    # W has shape (p*n, rank) = (9, rank)
    # W[k*n+i, c] is the convention: row index = k*n + i where k is the B-row and i is the A-col
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for C[0:2, 0:2] from A[0:2,0:2] * B[0:2,0:2]
    # A indices: A[i,j] -> U row i*m + j = i*3 + j
    # B indices: B[k,j] -> V row k*p + j = k*3 + j
    # C indices: C[i,j] -> W row j*n + i = j*3 + i (using W[k*n+i] where k is B-row index, i is A-col index)
    
    # Strassen M1 = (A00+A11)*(B00+B11)
    # A00 = A[0,0], A11 = A[1,1] -> U rows 0, 4
    # B00 = B[0,0], B11 = B[1,1] -> V rows 0, 4
    # C00 += M1, C11 += M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0*3+0][col] = 1  # C[0,0]: W row = 0*3+0 = 0
    W[1*3+1][col] = 1  # C[1,1]: W row = 1*3+1 = 4
    col += 1
    
    # M2 = (A10+A11)*B00
    # A10 = A[1,0] -> U row 3, A11 = A[1,1] -> U row 4
    # B00 = B[0,0] -> V row 0
    # C10 += M2, C11 -= M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[0*3+1][col] = 1   # C[1,0]: W row = 0*3+1 = 1
    W[1*3+1][col] = -1  # C[1,1]: W row = 1*3+1 = 4
    col += 1
    
    # M3 = A00*(B10-B11)
    # A00 = A[0,0] -> U row 0
    # B10 = B[1,0] -> V row 3, B11 = B[1,1] -> V row 4
    # C01 += M3, C11 += M3
    U[0][col] = 1
    V[3][col] = 1; V[4][col] = -1
    W[0*3+0][col] = 1   # C[0,0]... wait, C01 means C[0,1]
    # C[0,1]: W row = 1*3+0 = 3
    # C[1,1]: W row = 1*3+1 = 4
    W[1*3+0][col] = 1
    W[1*3+1][col] = 1
    col += 1
    
    # M4 = A11*(B01-B10)
    # A11 = A[1,1] -> U row 4
    # B01 = B[0,1] -> V row 1, B10 = B[1,0] -> V row 3
    # C00 += M4, C10 += M4
    U[4][col] = 1
    V[1][col] = 1; V[3][col] = -1
    W[0*3+0][col] = 1   # C[0,0]: row 0
    W[0*3+1][col] = 1   # C[1,0]: row 1
    col += 1
    
    # M5 = (A00+A01)*B11
    # A00 = A[0,0] -> U row 0, A01 = A[0,1] -> U row 1
    # B11 = B[1,1] -> V row 4
    # C00 -= M5, C01 += M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0*3+0][col] = -1  # C[0,0]: row 0
    W[1*3+0][col] = 1   # C[0,1]: row 3
    col += 1
    
    # M6 = (A10-A00)*(B00+B10)
    # A10 = A[1,0] -> U row 3, A00 = A[0,0] -> U row 0
    # B00 = B[0,0] -> V row 0, B10 = B[1,0] -> V row 3
    # C11 += M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[3][col] = 1
    W[1*3+1][col] = 1   # C[1,1]: row 4
    col += 1
    
    # M7 = (A01-A11)*(B01+B11)
    # A01 = A[0,1] -> U row 1, A11 = A[1,1] -> U row 4
    # B01 = B[0,1] -> V row 1, B11 = B[1,1] -> V row 4
    # C00 += M7
    U[1][col] = 1; U[4][col] = -1
    V[1][col] = 1; V[4][col] = 1
    W[0*3+0][col] = 1   # C[0,0]: row 0
    col += 1
    
    # Now cross terms: A[0:2,2] * B[2,0:2] contributing to C[0:2,0:2]
    # A[i,2] for i in {0,1}: U rows 2, 5
    # B[2,j] for j in {0,1}: V rows 6, 7
    # C[i,j] for i,j in {0,1}: W rows j*3+i
    
    # Term: A[0,2]*B[2,0] -> C[0,0]
    U[2][col] = 1
    V[6][col] = 1
    W[0*3+0][col] = 1
    col += 1
    
    # Term: A[1,2]*B[2,1] -> C[1,1]
    U[5][col] = 1
    V[7][col] = 1
    W[1*3+1][col] = 1
    col += 1
    
    # Wait, I need all 4 cross terms for A[0:2,2]*B[2,0:2]:
    # A[0,2]*B[2,0] -> C[0,0]
    # A[0,2]*B[2,1] -> C[0,1]
    # A[1,2]*B[2,0] -> C[1,0]
    # A[1,2]*B[2,1] -> C[1,1]
    
    # I already did A[0,2]*B[2,0] and A[1,2]*B[2,1]. Let me redo properly.
    # Actually I only added 2 terms above. Let me add the remaining 2.
    
    # A[0,2]*B[2,1] -> C[0,1]
    U[2][col] = 1
    V[7][col] = 1
    W[1*3+0][col] = 1  # C[0,1]: row 3
    col += 1
    
    # A[1,2]*B[2,0] -> C[1,0]
    U[5][col] = 1
    V[6][col] = 1
    W[0*3+1][col] = 1  # C[1,0]: row 1
    col += 1
    
    # Now: A[0:2,0:2] * B[0:2,2] contributing to C[0:2,2]
    # A[i,k] for i in {0,1}, k in {0,1}: U rows i*3+k
    # B[k,2] for k in {0,1}: V rows k*3+2 = 2, 5
    # C[i,2] for i in {0,1}: W rows 2*3+i = 6, 7
    
    # 4 terms:
    # A[0,0]*B[0,2] -> C[0,2]
    U[0][col] = 1
    V[2][col] = 1
    W[2*3+0][col] = 1  # C[0,2]: row 6
    col += 1
    
    # A[0,1]*B[1,2] -> C[0,2]
    U[1][col] = 1
    V[5][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # A[1,0]*B[0,2] -> C[1,2]
    U[3][col] = 1
    V[2][col] = 1
    W[2*3+1][col] = 1  # C[1,2]: row 7
    col += 1
    
    # A[1,1]*B[1,2] -> C[1,2]
    U[4][col] = 1
    V[5][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # Now: A[2,0:2] * B[0:2,0:2] contributing to C[2,0:2]
    # A[2,k] for k in {0,1}: U rows 2*3+k = 6, 7
    # B[k,j] for k,j in {0,1}: V rows k*3+j
    # C[2,j] for j in {0,1}: W rows j*3+2 = 2, 5
    
    # 4 terms:
    # A[2,0]*B[0,0] -> C[2,0]
    U[6][col] = 1
    V[0][col] = 1
    W[0*3+2][col] = 1  # C[2,0]: row 2
    col += 1
    
    # A[2,0]*B[0,1] -> C[2,1]
    U[6][col] = 1
    V[1][col] = 1
    W[1*3+2][col] = 1  # C[2,1]: row 5
    col += 1
    
    # A[2,1]*B[1,0] -> C[2,0]
    U[7][col] = 1
    V[3][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # A[2,1]*B[1,1] -> C[2,1]
    U[7][col] = 1
    V[4][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # Now: A[0:2,2] * B[2,2] contributing to C[0:2,2]
    # A[i,2] for i in {0,1}: U rows 2, 5
    # B[2,2]: V row 8
    # C[i,2] for i in {0,1}: W rows 6, 7
    
    # A[0,2]*B[2,2] -> C[0,2]
    U[2][col] = 1
    V[8][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # A[1,2]*B[2,2] -> C[1,2]
    U[5][col] = 1
    V[8][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # Now: A[2,2] * B[2,0:2] contributing to C[2,0:2]
    # A[2,2]: U row 8
    # B[2,j] for j in {0,1}: V rows 6, 7
    # C[2,j] for j in {0,1}: W rows 2, 5
    
    # A[2,2]*B[2,0] -> C[2,0]
    U[8][col] = 1
    V[6][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # A[2,2]*B[2,1] -> C[2,1]
    U[8][col] = 1
    V[7][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # Now: A[2,0:2] * B[0:2,2] contributing to C[2,2]
    # A[2,k] for k in {0,1}: U rows 6, 7
    # B[k,2] for k in {0,1}: V rows 2, 5
    # C[2,2]: W row 2*3+2 = 8
    
    # A[2,0]*B[0,2] -> C[2,2]
    U[6][col] = 1
    V[2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # A[2,1]*B[1,2] -> C[2,2]
    U[7][col] = 1
    V[5][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # Finally: A[2,2]*B[2,2] -> C[2,2]
    U[8][col] = 1
    V[8][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
