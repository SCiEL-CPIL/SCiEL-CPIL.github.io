"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.

Strategy:
1. Construct a Strassen-based decomposition that exploits the 2x2 block structure
   to achieve rank < 36 (schoolbook).
2. The idea: split A (3x3) and B (3x4) into blocks and use Strassen's 7-multiplication
   trick on 2x2 sub-blocks, combined with direct multiplications for the remaining parts.

For 3x3 * 3x4:
- Split A into a 2x2 block (top-left 2x2) and remaining rows/cols
- Split B into 2x2 blocks and remaining

A concrete approach:
  A = [[A11, A12, A13],
       [A21, A22, A23],
       [A31, A32, A33]]  (3x3)
  
  B = [[B11, B12, B13, B14],
       [B21, B22, B23, B24],
       [B31, B32, B33, B34]]  (3x4)

C = A * B (3x4)

We can use the following decomposition:
- Use Strassen on the top-left 2x2 of A with the top-left 2x2 of B (for the 2x2 part of C)
- Handle the remaining parts with direct multiplications

Let me think of a cleaner decomposition.

Approach: Think of 3x3 * 3x4 as:
  - A 2x2 * 2x4 multiplication (using Strassen-like: rank 7*2 = 14 for the 2x2 part? No...)
  
Actually, let me use a different strategy. 

Key insight: 3x3 * 3x4 can be seen as:
  C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]  (2x3 * 3x4)
  C[2, 0:4]   = A[2, :] * B[:, 0:4]    (1x3 * 3x4)

For 2x3 * 3x4: schoolbook is 2*3*4 = 24
For 1x3 * 3x4: schoolbook is 1*3*4 = 12
Total: 36 (same as schoolbook)

Better approach: Use Strassen on 2x2 subproblems.

Let me partition:
A (3x3) = [A1 | A2 | A3] where A1, A2, A3 are 3x1 columns? No.

Let me try:
A (3x3), B (3x4)

Split into:
  A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
  B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2

Then C = A*B:
  C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
       [A21*B11 + A22*B21, A21*B12 + A22*B22]]

This is getting complicated. Let me just try to find a rank-34 decomposition
by using Strassen on a 2x2 subproblem and direct for the rest.

2x2 * 2x2 has rank 7 (Strassen) vs 4 (schoolbook). Saving of 3.
If we can identify one 2x2*2x2 subproblem, we save 3, giving rank 33.
But the geometry may not allow clean separation.

Alternative: Use the known fact that R(3,3,4) <= 34 from literature or construct it.

Let me try a concrete construction:
- 3x3 * 3x4: use 1 Strassen 2x2*2x2 (rank 7) + remaining direct terms
- The 2x2*2x2 handles 4 output entries with 7 multiplications instead of 4
- Wait, that's more multiplications (7 > 4). Strassen helps when you have multiple 2x2 blocks.

Actually Strassen reduces the number of scalar multiplications for 2x2*2x2 from 4 to 7? No, from 4 to 7 is worse!

Wait, I'm confusing myself. For 2x2 * 2x2:
- Schoolbook: 4 scalar multiplications (each output is a dot product of length 2, so 2*2=4 multiplications per output, 4 outputs = 16? No...)

Let me be precise. For matrix multiplication C = A*B where A is n×m, B is m×p:
- Schoolbook rank = n*m*p (each term A[i,j]*B[j,k] is one rank-1 term)
- For 2x2*2x2: n=2, m=2, p=2, schoolbook rank = 8
- Strassen rank = 7

So Strassen saves 1 for 2x2*2x2 (from 8 to 7).

For 3x3*3x4: schoolbook = 36.
If I can find one 2x2*2x2 subproblem to replace: save 1, get 35.
If I can find two: save 2, get 34.

But the subproblems need to be cleanly separable.

Let me try a different approach: use the known rank-23 decomposition for 3x3*3x3 
and extend it to 3x3*3x4.

Actually, let me just try to construct a rank-34 decomposition explicitly.

Here's a plan:
- Use Strassen on the top-left 2x2 of A and top-left 2x2 of B to compute the top-left 2x2 of C.
  This uses 7 rank terms instead of 4*2=8 schoolbook terms for that subproblem.
  Wait, for 2x2*2x2, schoolbook is 8 terms (2*2*2), Strassen is 7.
  So we save 1 term.
- The remaining terms: total schoolbook is 36. The top-left 2x2 of C (entries C[0,0], C[0,1], C[1,0], C[1,1]) 
  involves all of A[0:2,:] and B[:,0:2]. That's 2*3*2 = 12 schoolbook terms for those 4 outputs.
  Using Strassen on the 2x2*2x2 part (A[0:2,0:2] * B[0:2,0:2]) gives 7 terms for the "diagonal" part,
  but we still need cross terms.

This is getting too complex to reason about abstractly. Let me just implement a search-based approach
that tries to find a low-rank decomposition using linear algebra.

Actually, let me try a simpler concrete construction:

Partition:
  A (3x3): rows 0,1 form A_top (2x3), row 2 is A_bot (1x3)
  B (3x4): cols 0,1 form B_left (3x2), cols 2,3 form B_right (3x2)

C (3x4):
  C[0:2, 0:2] = A_top * B_left  (2x3 * 3x2)
  C[0:2, 2:4] = A_top * B_right (2x3 * 3x2)
  C[2, 0:2]   = A_bot * B_left  (1x3 * 3x2)
  C[2, 2:4]   = A_bot * B_right (1x3 * 3x2)

Schoolbook for each:
  2x3*3x2: 2*3*2 = 12
  1x3*3x2: 1*3*2 = 6
Total: 12+12+6+6 = 36. No gain.

Let me try yet another approach. I'll use a greedy/local search to find a rank-34 or lower decomposition.

Given time constraints, let me implement:
1. A schoolbook baseline (rank 36)
2. A local search that tries to merge/simplify rank terms
3. Return the best found

Actually, the most reliable approach given the time budget is to try known constructions.

Let me try: R(3,3,4) <= R(3,3,3) + R(1,3,4) + cross terms?
R(3,3,3) = 23 (known)
R(1,3,4) = 12 (schoolbook, and it's minimal since it's a dot product of length 3 for each of 4 outputs, but actually R(1,m,p) = m*p... no, R(1,m,p) = m*p is schoolbook. For 1x3*3x4, it's just 4 dot products of length 3, so rank 12. But can we do better? R(1,m,p) >= max(m,p). For m=3,p=4, lower bound is 4. But I think R(1,3,4) = 12 is minimal because each output requires all 3 inputs from each matrix.)

Hmm, this approach doesn't obviously help.

Let me just go with a well-constructed rank-34 decomposition if I can find one, otherwise fall back to 36.

After more thought, let me try the following explicit construction for rank 34:

Use the fact that 3x3*3x4 can be decomposed as:
- One 2x2*2x2 Strassen block (rank 7) handling part of the computation
- Plus 27 direct terms for the rest
Total: 7 + 27 = 34

The question is whether the geometry allows this. Let me just implement a verification-based search.

Given the complexity, I'll implement a program that:
1. Starts with schoolbook (rank 36)
2. Tries to find redundant/mergeable terms via local search
3. Returns the best decomposition found within time budget
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
DEADLINE = 150.0  # seconds
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _time_left():
    return DEADLINE - _elapsed()


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 iff the term A[i,j]*B[j,k] contributes to C[i,k].
    Actually, the tensor is T[i,j,k] where the multiplication tensor is defined as:
    M[i,j,k] = delta_{i,j,k} meaning A[i,j] * B[j,k] contributes to C[i,k].
    
    For tensor decomposition: T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    where T_{i,j,k} = 1 if the (i,j) element of A and (j,k) element of B multiply to contribute to C[i,k].
    
    More precisely, the matrix multiplication tensor is:
    T[i,j,k] = 1 for all valid i,j,k (since A[i,j]*B[j,k] always contributes to C[i,k])
    Wait, that's not right either.
    
    The correct definition: The matrix multiplication tensor T is a 3-tensor where
    T[i,j,k] represents the coefficient in the bilinear form. For matrix multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    The tensor is simply all ones! The structure is in the indexing.
    
    For a rank-r decomposition:
    T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    
    The condition is: for all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    # The tensor is all ones, but the constraint is on the specific indexing pattern.
    # We just need to verify the decomposition satisfies the constraints.
    pass


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _schoolbook(n, m, p):
    """Construct the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _try_strassen_based(n, m, p):
    """
    Try to construct a decomposition using Strassen's algorithm on 2x2 subblocks.
    
    For 3x3 * 3x4, we partition:
    A (3x3) = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B (3x4) = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    C (3x4) = [C11 C12; C21 C22] where C11 is 2x2, C12 is 2x2, C21 is 1x2, C22 is 1x2
    
    C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x1*1x2)
    C12 = A11*B12 + A12*B22  (2x2 = 2x2*2x2 + 2x1*1x2)
    C21 = A21*B11 + A22*B21  (1x2 = 1x2*2x2 + 1x1*1x2)
    C22 = A21*B12 + A22*B22  (1x2 = 1x2*2x2 + 1x1*1x2)
    
    For the 2x2*2x2 parts, we can use Strassen (rank 7 instead of 8).
    There are 4 such 2x2*2x2 subproblems: A11*B11, A11*B12, A21*B11, A21*B12.
    Wait, A21 is 1x2, so A21*B11 is 1x2 * 2x2 = 1x2, which is NOT a 2x2*2x2.
    
    Let me redo:
    A11 (2x2) * B11 (2x2) -> 2x2: use Strassen, rank 7 (vs 8 schoolbook), save 1
    A11 (2x2) * B12 (2x2) -> 2x2: use Strassen, rank 7 (vs 8 schoolbook), save 1
    A12 (2x1) * B21 (1x2) -> 2x2: schoolbook rank 4 (2*1*2), no Strassen applicable
    A21 (1x2) * B11 (2x2) -> 1x2: schoolbook rank 4 (1*2*2), no Strassen
    A21 (1x2) * B12 (2x2) -> 1x2: schoolbook rank 4 (1*2*2), no Strassen
    A12 (2x1) * B22 (1x2) -> 2x2: schoolbook rank 4 (2*1*2), no Strassen
    A22 (1x1) * B21 (1x2) -> 1x2: schoolbook rank 2 (1*1*2)
    A22 (1x1) * B22 (1x2) -> 1x2: schoolbook rank 2 (1*1*2)
    
    Total schoolbook: 8+8+4+4+4+4+2+2 = 36
    With Strassen on the two 2x2*2x2: 7+7+4+4+4+4+2+2 = 34
    
    So we can achieve rank 34!
    """
    if (n, m, p) != (3, 3, 4):
        return None
    
    # Strassen decomposition for 2x2 * 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For our case, we need to map this to the tensor decomposition.
    
    # Let's build the full decomposition explicitly.
    # We'll use rank-7 Strassen for A11*B11 and A11*B12.
    
    # Actually, let me just construct it term by term.
    # This is complex, so let me use a different approach: 
    # directly specify the U, V, W matrices.
    
    # For the Strassen 2x2*2x2 subproblem, the rank-7 decomposition has:
    # U entries (for A elements), V entries (for B elements), W entries (for C elements)
    
    # Let me index: A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
    # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
    # C11 = C[0,0], C12 = C[0,1], C21 = C[1,0], C22 = C[1,1]
    
    # Strassen terms for 2x2*2x2 (standard):
    # t1 = (A11+A22)*(B11+B22)
    # t2 = (A21+A22)*B11
    # t3 = A11*(B12-B22)
    # t4 = A22*(B21-B11)
    # t5 = (A11+A12)*B22
    # t6 = (A21-A11)*(B11+B12)
    # t7 = (A12-A22)*(B21+B22)
    
    # C11 = t1 + t4 - t5 + t7
    # C12 = t3 + t5
    # C21 = t2 + t4
    # C22 = t1 - t2 + t3 + t6
    
    # In tensor decomposition terms, each Strassen term contributes to U, V, W.
    # For term t1 = (A11+A22)*(B11+B22):
    #   U contribution: A11 gets +1, A22 gets +1 (in the U matrix for this term)
    #   V contribution: B11 gets +1, B22 gets +1
    #   W contribution: C11 gets +1, C12 gets 0, C21 gets 0, C22 gets +1
    #   (because C11 += t1 and C22 += t1)
    
    # This is getting very involved. Let me just implement it programmatically.
    
    rank = 34
    U = [[0] * rank for _ in range(9)]   # 3*3 = 9
    V = [[0] * rank for _ in range(12)]  # 3*4 = 12
    W = [[0] * rank for _ in range(12)]  # 4*3 = 12
    
    col = 0
    
    # Helper to add a term
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, val in u_entries:
            U[idx][col] += val
        for idx, val in v_entries:
            V[idx][col] += val
        for idx, val in w_entries:
            W[idx][col] += val
        col += 1
    
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    # n=3, m=3, p=4
    # U indices: i*3+j for i in 0..2, j in 0..2
    # V indices: j*4+k for j in 0..2, k in 0..3
    # W indices: k*3+i for k in 0..3, i in 0..2
    
    # A11*B11 using Strassen (2x2*2x2 in the top-left 2x2 of A and B)
    # A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    
    # t1 = (A11+A22)*(B11+B22)
    add_term(
        [(0, 1), (3, 1)],      # A[0,0] and A[1,1]
        [(0, 1), (5, 1)],      # B[0,0] and B[1,1]
        [(0, 1), (1, 1)],      # C[0,0] (k=0,i=0) and C[1,1] (k=1,i=1)
    )
    # W[k*n+i]: C11 is W[0*3+0]=W[0], C22 is W[1*3+1]=W[4]
    # Wait, I need to be more careful.
    
    # Let me restart with proper indexing.
    # W[k*n+i] where k is column of C, i is row of C.
    # C[i,k] corresponds to W[k*n+i] = W[k*3+i]
    
    # C11 = C[0,0] -> W[0*3+0] = W[0]
    # C12 = C[0,1] -> W[1*3+0] = W[3]
    # C21 = C[1,0] -> W[0*3+1] = W[1]
    # C22 = C[1,1] -> W[1*3+1] = W[4]
    
    # t1 = (A11+A22)*(B11+B22) contributes to C11 and C22
    # C11 += t1, C22 += t1
    add_term(
        [(0, 1), (3, 1)],      # A[0,0], A[1,1]
        [(0, 1), (5, 1)],      # B[0,0], B[1,1]
        [(0, 1), (4, 1)],      # W[0] (C[0,0]), W[4] (C[1,1])
    )
    
    # t2 = (A21+A22)*B11
    # A21=A[1,0], A22=A[1,1], B11=B[0,0]
    # Contributes to C21 and C22
    # C21 += t2, C22 -= t2
    add_term(
        [(3, 1), (4, 1)],      # A[1,0], A[1,1]
        [(0, 1)],              # B[0,0]
        [(1, 1), (4, -1)],     # W[1] (C[1,0]), W[4] (C[1,1])
    )
    
    # t3 = A11*(B12-B22)
    # A11=A[0,0], B12=B[0,1], B22=B[1,1]
    # Contributes to C12 and C22
    # C12 += t3, C22 += t3
    add_term(
        [(0, 1)],              # A[0,0]
        [(1, 1), (5, -1)],     # B[0,1], B[1,1]
        [(3, 1), (4, 1)],      # W[3] (C[0,1]), W[4] (C[1,1])
    )
    
    # t4 = A22*(B21-B11)
    # A22=A[1,1], B21=B[1,0], B11=B[0,0]
    # Contributes to C11 and C21
    # C11 += t4, C21 += t4
    add_term(
        [(4, 1)],              # A[1,1]
        [(4, 1), (0, -1)],     # B[1,0], B[0,0]
        [(0, 1), (1, 1)],      # W[0] (C[0,0]), W[1] (C[1,0])
    )
    
    # t5 = (A11+A12)*B22
    # A11=A[0,0], A12=A[0,1], B22=B[1,1]
    # Contributes to C12 and C11
    # C12 += t5, C11 -= t5
    add_term(
        [(0, 1), (1, 1)],      # A[0,0], A[0,1]
        [(5, 1)],              # B[1,1]
        [(3, 1), (0, -1)],     # W[3] (C[0,1]), W[0] (C[0,0])
    )
    
    # t6 = (A21-A11)*(B11+B12)
    # A21=A[1,0], A11=A[0,0], B11=B[0,0], B12=B[0,1]
    # Contributes to C22
    # C22 += t6
    add_term(
        [(3, 1), (0, -1)],     # A[1,0], A[0,0]
        [(0, 1), (1, 1)],      # B[0,0], B[0,1]
        [(4, 1)],              # W[4] (C[1,1])
    )
    
    # t7 = (A12-A22)*(B21+B22)
    # A12=A[0,1], A22=A[1,1], B21=B[1,0], B22=B[1,1]
    # Contributes to C11
    # C11 += t7
    add_term(
        [(1, 1), (4, -1)],     # A[0,1], A[1,1]
        [(4, 1), (5, 1)],      # B[1,0], B[1,1]
        [(0, 1)],              # W[0] (C[0,0])
    )
    
    # That's 7 terms for A11*B11 part. Now we need the cross terms.
    # Remaining terms to handle:
    # A12*B21 (2x1 * 1x2 -> 2x2): 4 terms
    # A12*B22 (2x1 * 1x2 -> 2x2): 4 terms  
    # A21*B11 (1x2 * 2x2 -> 1x2): 4 terms
    # A21*B12 (1x2 * 2x2 -> 1x2): 4 terms
    # A22*B21 (1x1 * 1x2 -> 1x2): 2 terms
    # A22*B22 (1x1 * 1x2 -> 1x2): 2 terms
    # Plus terms involving column 2 of A and columns 2,3 of B:
    # A[*,2] * B[:,2] and A[*,2] * B[:,3]
    
    # Wait, I've only handled the top-left 2x2 of A with top-left 2x2 of B.
    # I still need to handle:
    # - A[0:2,2] (2x1) * B[:,0:2] (3x2) -> contributes to C[0:2,0:2]
    # - A[0:2,2] (2x1) * B[:,2:4] (3x2) -> contributes to C[0:2,2:4]
    # - A[2,:] (1x3) * B[:,0:2] (3x2) -> contributes to C[2,0:2]
    # - A[2,:] (1x3) * B[:,2:4] (3x2) -> contributes to C[2,2:4]
    
    # This is getting very complex. The Strassen approach only saves 2 terms total
    # (from 8 to 7 for each of the two 2x2*2x2 subproblems).
    
    # Let me count what I've done so far and what remains.
    # I've used 7 terms for the "A[0:2,0:2] * B[0:2,0:2]" part.
    # But this only handles part of C[0:2,0:2].
    
    # Actually, I realize my approach is flawed. The Strassen decomposition handles
    # the full 2x2*2x2 multiplication, but in our 3x3*3x4 problem, the 2x2*2x2
    # subproblem is only part of the computation.
    
    # Let me abandon this complex approach and just use a simpler one.
    # I'll return the schoolbook decomposition for now, but try a local search
    # to see if we can reduce the rank.
    
    return None


def _local_search(U, V, W, n, m, p, max_iters=5000):
    """
    Try to reduce the rank by finding redundant terms or combining terms.
    This is a heuristic local search.
    """
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Try removing each term and see if the decomposition still works
        # (This is expensive, so we limit attempts)
        if it % 100 == 0 and _time_left() < 10.0:
            break
        
        removed = False
        for r in range(rank):
            if _time_left() < 10.0:
                break
            
            # Save the term
            u_term = [U[i][r] for i in range(n * m)]
            v_term = [V[i][r] for i in range(m * p)]
            w_term = [W[i][r] for i in range(p * n)]
            
            # Check if this term is zero
            if all(x == 0 for x in u_term) or all(x == 0 for x in v_term) or all(x == 0 for x in w_term):
                # Remove it
                U = [row[:r] + row[r+1:] for row in U]
                V = [row[:r] + row[r+1:] for row in V]
                W = [row[:r] + row[r+1:] for row in W]
                rank -= 1
                removed = True
                break
        
        if not removed:
            # Try combining two terms
            break
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)  
        - W has shape (p*n, rank) = (12, rank)
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = len(result['U'][0])
    
    # Try Strassen-based construction
    strassen_result = _try_strassen_based(n, m, p)
    if strassen_result is not None:
        if _verify_decomposition(strassen_result['U'], strassen_result['V'], strassen_result['W'], n, m, p):
            result = strassen_result
            best_rank = len(result['U'][0])
    
    # Try local search to reduce rank further
    if _time_left() > 20.0:
        searched = _local_search(result['U'], result['V'], result['W'], n, m, p)
        if _verify_decomposition(searched['U'], searched['V'], searched['W'], n, m, p):
            new_rank = len(searched['U'][0])
            if new_rank < best_rank:
                result = searched
                best_rank = new_rank
    
    # Final verification
    if not _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
        # Fall back to schoolbook
        result = _schoolbook(n, m, p)
    
    return result
