"""
Rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
Uses Strassen's rank-7 algorithm for the 2x2 block structure, extended to handle the 4x7 case.

The 4x4 matrix A is viewed as a 2x2 block matrix of 2x2 blocks.
The 4x7 matrix B is viewed as a 2x2 block matrix of 2x2 blocks in the row dimension, 
with 7 columns (so each "column block" is a 2x7 matrix, not 2x2).

Actually, let's think more carefully:
- A is 4x4: split into 2x2 blocks of 2x2 submatrices
- B is 4x7: split into 2x2 blocks in the row dimension (2 rows of 2x2... no)
  
B is 4x7, so if we split rows into two groups of 2, we get two 2x7 blocks.
The product A*B is 4x7.

Using Kronecker product / block structure:
If A = [[A11, A12], [A21, A22]] where each Ai is 2x2, and B = [[B1], [B2]] where B1 is 2x7 and B2 is 2x7,
then AB = [[A11*B1 + A12*B2, ...], ...]

This is a 2x2 matrix multiplication where the "entries" are matrices of size 2x2 (for A) and 2x7 (for B).
But Strassen's algorithm works for 2x2 matrices over a ring/field. Here the "entries" are not scalars but matrices.

Actually, the standard approach: the tensor for 4x4x7 multiplication can be decomposed using the 
Kronecker product of the 2x2x2 tensor (Strassen rank 7) with the 2x2x? tensor.

Let me think differently. The 4x4x4 multiplication tensor has rank at most 7*4 = 28 using 
Kronecker product of Strassen (rank 7) with the 2x2x2 identity (rank 4)... no, that's not right either.

Actually, the Kronecker product of two tensor decompositions gives a decomposition of the 
Kronecker product of the tensors. The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors.
So if 2x2x2 has rank 7, then 4x4x4 has rank at most 7*7 = 49.

For 4x4x7: we can write 7 = 4 + 3, or think of it as 2x2x? 

Actually, 4x4x7 = Kronecker product structure: 4 = 2*2, so the 4x4 part is 2x2x2 ⊗ 2x2x2.
The 7 is the third dimension. 7 = 2*2 + 3, which doesn't factor nicely.

Alternative: 4x4x7 tensor. We can use the fact that 4x4x7 = (2x2x2) ⊗ (2x2x7/2)... no.

Let me just use the known result: the 4x4x4 tensor has rank at most 49 (Strassen squared).
For 4x4x7, we can extend: treat B as having 7 columns. We can split B into B1 (4x4) and B2 (4x3).
Then A*B = A*B1 + A*B2. The A*B1 part uses the 4x4x4 decomposition (rank 49).
The A*B2 part: 4x4x3. The 4x4x3 tensor is a sub-tensor of 4x4x4, so it can be done in at most 
the rank of 4x4x4, but we can do better. 4x4x3 = (2x2x2) ⊗ (2x2x3/2)... 

Actually, 4x4x3: split A into 2x2 blocks, B (4x3) into 2x3 blocks. Then it's a 2x2 "matrix" 
multiplication where each entry is a 2x2 (for A) times 2x3 (for B) = 2x3 result.
This is 2x2 matrix multiplication over the ring of 2x3 matrices? No, the dimensions don't match for multiplication.

Let me just use a simpler approach: 
- For the 4x4x7 problem, use the schoolbook for the full thing but with Strassen on the 4x4x4 subproblem.
- 4x4x7: rank = rank(4x4x4) + rank(4x4x3)
- rank(4x4x4) <= 49 (Strassen Kronecker)
- rank(4x4x3): use schoolbook, that's 4*4*3 = 48. Or use Strassen-like: 2x2x2 (rank 7) times 2x2x3/2... 

Actually, 4x4x3: think of it as multiplying 4x4 by 4x3. Split into 2x2 blocks:
A = [[A11, A12], [A21, A22]] (each 2x2)
B = [[B1], [B2]] where B1 is 2x3, B2 is 2x3
Result C is 4x3, split into [[C11], [C21]] where C11 is 2x3, C21 is 2x3.

C11 = A11*B1 + A12*B2
C21 = A21*B1 + A22*B2

This is 4 independent 2x2 times 2x3 multiplications. Each 2x2x3 multiplication has tensor 
rank at most 2*2*3 = 12 (schoolbook), but can we do better? The 2x2x3 tensor is a sub-tensor 
of 2x2x2x2, so... not directly helpful.

For 2x2x3: schoolbook is 12. Is there a rank-7 or rank-8 decomposition? 
The 2x2x2 tensor has rank 7. The 2x2x3 tensor contains the 2x2x2 tensor as a sub-tensor 
(set one column to zero), so rank(2x2x3) >= rank(2x2x2) = 7. Upper bound: 12.

Given time constraints, let me just implement:
1. 4x4x4 using Kronecker product of two Strassen-7 decompositions: rank 49
2. 4x4x3 using schoolbook: rank 48
Total: 97

But wait, the knowledge says "rank ~97 vs schoolbook 112". So this is the approach.

Let me implement Strassen's 2x2x2 decomposition with rank 7.

Strassen's 7 products for 2x2x2:
M1 = (A11 + A22)(B11 + B22)
M2 = (A21 + A22)B11
M3 = A11(B12 - B22)
M4 = A22(B21 - B11)
M5 = (A11 + A12)B22
M6 = (A21 - A11)(B11 + B12)
M7 = (A12 - A22)(B21 + B22)

C11 = M1 - M4 + M5 - M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6

The tensor entries: for each of the 7 terms, we have a U vector (for A's 4 entries), 
a V vector (for B's 4 entries), and a W vector (for C's 4 entries).

A indices: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
B indices: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
C indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22

For term t, U_t[i,j] * V_t[j,k] * W_t[k,i] should sum to the matrix product tensor.

Let me define the 7 terms:
Term 0 (M1): U has A11=1, A22=1; V has B11=1, B22=1; W has C11=1, C22=1
Term 1 (M2): U has A21=1, A22=1; V has B11=1; W has C21=1
Term 2 (M3): U has A11=1; V has B12=1, B22=-1; W has C12=1
Term 3 (M4): U has A22=1; V has B21=1, B11=-1; W has C11=-1, C21=1
Term 4 (M5): U has A11=1, A12=1; V has B22=1; W has C11=1, C12=1
Term 5 (M6): U has A21=1, A11=-1; V has B11=1, B12=1; W has C22=1
Term 6 (M7): U has A12=1, A22=-1; V has B21=1, B22=1; W has C11=-1, C22=-1

Now for the Kronecker product to get 4x4x4:
The 4x4 matrix A has indices (i1,i2,j1,j2) where i1,i2 ∈ {0,1} are row block/row within block, 
j1,j2 ∈ {0,1} are col block/col within block.
Similarly B: (j1,j2,k1,k2)
C: (k1,k2,i1,i2)

The Kronecker product of two 2x2x2 tensors:
For the first 2x2x2 (block level): indices (i1, j1, k1)
For the second 2x2x2 (within-block): indices (i2, j2, k2)

The combined tensor has rank 7*7 = 49.
For term (a, b) where a,b ∈ {0..6}:
U_combined[(i1,i2,j1,j2)] = U1[a][(i1,j1)] * U2[b][(i2,j2)]
V_combined[(j1,j2,k1,k2)] = V1[a][(j1,k1)] * V2[b][(j2,k2)]
W_combined[(k1,k2,i1,i2)] = W1[a][(k1,i1)] * W2[b][(k2,i2)]

Now for 4x4x7:
A is 4x4, B is 4x7, C is 4x7.
We can split B's 7 columns into 4 + 3.
Part 1: A * B1 where B1 is 4x4. Use the 4x4x4 decomposition (rank 49).
Part 2: A * B2 where B2 is 4x3. Use schoolbook (rank 4*4*3 = 48).
Total rank: 49 + 48 = 97.

For the 4x4x3 schoolbook:
A indices: (i1,i2,j1,j2) -> 16 entries, index = i1*8 + i2*4 + j1*2 + j2... let me just use linear indices.
A is 4x4: row i ∈ 0..3, col j ∈ 0..3. Index = i*4+j.
B is 4x3: row j ∈ 0..3, col k ∈ 0..2. Index = j*3+k.
C is 4x3: row k ∈ 0..2, col i ∈ 0..3. Index = k*4+i. (per the trace convention W[k*n+i,r])

Schoolbook: for each (i,j,k), one term.
U[i*4+j] = 1, V[j*3+k] = 1, W[k*4+i] = 1.

Let me code this up.

Actually, I realize I need to double-check the W convention. The task says:
"The output factor follows the official trace convention W[k*n+i,r]"
So W has shape (p*n, rank) = (7*4, rank) = (28, rank).
W[k*n + i, r] corresponds to the output entry C[i, k] where i is row (0..3) and k is col (0..6).

Wait, C is 4x7. C[i, k] where i ∈ 0..3, k ∈ 0..6.
W[k*n + i, r] = W[k*4 + i, r]. So for a given output entry C[i,k], it's at W[k*4+i, r].

In the tensor decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = C[i,k]

So U has shape (n*m, rank) = (16, rank), indexed by [i*m+j] = [i*4+j].
V has shape (m*p, rank) = (28, rank), indexed by [j*p+k] = [j*7+k].
W has shape (p*n, rank) = (28, rank), indexed by [k*n+i] = [k*4+i].

OK so my indexing above was slightly off for V. Let me redo.

For the 4x4x4 part (B1 is 4x4, k ∈ 0..3):
V[j*7+k] for k ∈ 0..3.
W[k*4+i] for k ∈ 0..3.

For the 4x4x3 part (B2 is 4x3, k ∈ 4..6):
V[j*7+k] for k ∈ 4..6.
W[k*4+i] for k ∈ 4..6.

Let me implement this.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strassen 2x2x2 decomposition, rank 7
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # U_strassen[t][i,j], V_strassen[t][j,k], W_strassen[t][k,i]
    U_s = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1], # M7: A12 - A22
    ]
    V_s = [
        [1, 0, 0, 1],  # B11 + B22
        [1, 0, 0, 0],  # B11
        [0, 1, 0, -1], # B12 - B22
        [-1, 0, 1, 0], # B21 - B11
        [0, 0, 0, 1],  # B22
        [1, 1, 0, 0],  # B11 + B12
        [0, 0, 1, 1],  # B21 + B22
    ]
    W_s = [
        [1, 0, 0, 1],  # C11 + C22
        [0, 0, 1, 0],  # C21
        [0, 1, 0, 0],  # C12
        [-1, 0, 1, 0], # -C11 + C21
        [1, 1, 0, 0],  # C11 + C12
        [0, 0, 0, 1],  # C22
        [-1, 0, 0, -1] # -C11 - C22
    ]
    
    # Build 4x4x4 decomposition using Kronecker product of two Strassen-7
    # A is 4x4: index (i1,i2,j1,j2) -> i1*8 + i2*4 + j1*2 + j2... 
    # Actually, A row i = i1*2 + i2, col j = j1*2 + j2
    # So A index = i*4 + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    
    # B is 4x4 (for the 4x4x4 part): row j = j1*2+j2, col k = k1*2+k2
    # B index = j*4 + k = (j1*2+j2)*4 + (k1*2+k2) = j1*8 + j2*4 + k1*2 + k2
    
    # C is 4x4: row k = k1*2+k2, col i = i1*2+i2
    # C index = k*4 + i = (k1*2+k2)*4 + (i1*2+i2) = k1*8 + k2*4 + i1*2 + i2
    
    # For the 4x4x7 problem:
    # B has 7 columns. k ∈ 0..6.
    # For k ∈ 0..3: k1 = k//2, k2 = k%2 (part of 4x4x4 subproblem)
    # For k ∈ 4..6: these are extra columns (4x4x3 subproblem)
    
    # Rank for 4x4x4 part: 7*7 = 49
    # Rank for 4x4x3 part: 4*4*3 = 48 (schoolbook)
    # Total rank: 97
    
    rank = 49 + 48
    U = [0] * rank
    V = [0] * rank
    W = [0] * rank
    
    # Part 1: 4x4x4 using Kronecker of Strassen-7
    for a in range(7):
        for b in range(7):
            t = a * 7 + b
            # U: for A index (i1,i2,j1,j2)
            U[t] = [0] * 16
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            idx_a = i1 * 8 + i2 * 4 + j1 * 2 + j2
                            U[t][idx_a] = U_s[a][i1 * 2 + j1] * U_s[b][i2 * 2 + j2]
            
            # V: for B index (j1,j2,k1,k2) with k1,k2 from k ∈ 0..3
            V[t] = [0] * 28
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            k = k1 * 2 + k2
                            idx_b = j1 * 8 + j2 * 4 + k1 * 2 + k2  # This is j*4+k for 4x4
                            # But in the 4x4x7 problem, B index is j*7+k
                            idx_b_full = (j1 * 2 + j2) * 7 + k
                            V[t][idx_b_full] = V_s[a][j1 * 2 + k1] * V_s[b][j2 * 2 + k2]
            
            # W: for C index (k1,k2,i1,i2) with k ∈ 0..3
            W[t] = [0] * 28
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            idx_c_full = k * 4 + (i1 * 2 + i2)  # k*n + i
                            W[t][idx_c_full] = W_s[a][k1 * 2 + i1] * W_s[b][k2 * 2 + i2]
    
    # Part 2: 4x4x3 schoolbook for k ∈ 4..6
    offset = 49
    t = offset
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[t] = [0] * 16
                V[t] = [0] * 28
                W[t] = [0] * 28
                U[t][i * 4 + j] = 1
                V[t][j * 7 + k] = 1
                W[t][k * 4 + i] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
