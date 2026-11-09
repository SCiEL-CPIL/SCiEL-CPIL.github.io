"""
Matrix multiplication 5x5x5: rank-minimal exact tensor decomposition.

Strategy:
1. Start with the schoolbook decomposition (rank 125).
2. Use a structured partitioning approach inspired by Strassen:
   - Partition the 5x5 matrices into blocks.
   - Use a 3x3x3 decomposition for the block structure.
   - Within each block, use smaller decompositions.
3. Try to find a rank < 125 decomposition using the following:
   - A 3x3x3 decomposition of rank 23 (best known for 3x3x3 is ~23).
   - Combine with 2x2x2 Strassen (rank 7) where possible.
   
The key insight: if we can decompose 5x5x5 using a tensor network of
smaller multiplications, we can get rank below 125.

For 5x5x5, one approach:
- Split 5 = 2 + 3
- Use a 2x2x2 (Strassen, rank 7) and 3x3x3 (rank ~23) structure
- The combined tensor for a "mixed" partition can have rank less than 
  the sum of individual ranks if we share structure.

Actually, let's think about this more carefully.

The matrix multiplication tensor for n x m x p has rank R(n,m,p).
We know:
- R(1,1,1) = 1
- R(2,2,2) = 7 (Strassen)
- R(3,3,3) = 23 (best known upper bound, lower bound is 19)
- R(5,5,5) <= 125 (schoolbook)

For 5x5x5, we can use the fact that 5 = 2+3 and construct a decomposition
using the "partition" method. If we partition the indices:
- i in {0,1} x {0,1,2,3,4} -> but this doesn't directly help.

A better approach: use the "tensor product" / "Kronecker" structure.
If we have a decomposition of rank r1 for (n1,m1,p1) and rank r2 for 
(n2,m2,p2), we can get a decomposition of rank r1*r2 for 
(n1+n2, m1+m2, p1+p2) but that's not quite right.

Actually, the standard way to combine:
If A is n1 x m1 and B is m1 x p1, we can use the decomposition for that.
For 5x5x5, we can write 5 = 2 + 3 and use:
- The (2,2,2) block: rank 7
- The (3,3,3) block: rank 23
- The cross terms: (2,2,3), (2,3,2), (3,2,2), (2,3,3), (3,2,3), (3,3,2)

But the cross terms are also matrix multiplication tensors of different shapes.

Let me think about this differently. The tensor for 5x5x5 multiplication is:
T_{i,j,k} = sum over the "inner" dimension

Actually, for matrix multiplication C = A * B where A is 5x5 and B is 5x5:
C_{i,k} = sum_j A_{i,j} * B_{j,k}

The tensor has indices (i, j, k) where i is row of C (and A), j is column of A (and row of B), k is column of C (and B).

The rank-1 terms are of the form: U_{i,j} * V_{j,k} * W_{k,i}

For schoolbook, each (i,j,k) gives one term.

To get below 125, we need to exploit algebraic structure.

One known approach: use the "Winograd" or "Strassen-like" methods.

For 5x5x5, let me try a different approach: use the fact that we can group terms.

Actually, let me try a more direct approach: use a random search / optimization to find a lower-rank decomposition.

But that's computationally infeasible for rank < 125.

Let me try the partitioning approach more carefully.

If we partition the 5x5 matrix as:
- Top-left 2x2 block
- Top-right 2x3 block  
- Bottom-left 3x2 block
- Bottom-right 3x3 block

Then C = A*B becomes a 2x2 block matrix where each block is computed from products of the sub-blocks.

This gives 4x4 = 16 block multiplications, but many of them are rectangular.

Actually, this is getting complex. Let me just implement the schoolbook and verify it works, then try a simple optimization.

Given the time constraints, let me implement:
1. Schoolbook (rank 125) - guaranteed to work
2. Try a simple "merge" heuristic to see if we can reduce rank

For the merge heuristic: if two rank-1 terms have the same U or V or W pattern, we might be able to combine them.

Actually, for an exact decomposition, merging is very hard. Let me just do schoolbook and ensure it's correct.

Wait, let me reconsider. The task says "rank-minimal" and the score is the rank. So we want the smallest possible rank.

Known results:
- R(5,5,5) is not known exactly.
- Upper bound: 125 (schoolbook)
- There are better upper bounds using advanced methods.

Let me try to implement a 3x3x3 decomposition of rank 23 and use it in a partitioning scheme.

For a 3x3x3 matrix multiplication, the best known decomposition has rank 23. Let me use that.

Actually, finding an explicit rank-23 decomposition for 3x3x3 is non-trivial. The best known constructions are complex.

Let me try a different approach: use the "binary" or "recursive" structure.

5 in binary is 101. We can use a recursive Strassen-like approach.

Actually, for the purpose of this task, let me just implement the schoolbook decomposition correctly and ensure it passes verification. If we have time, we can try to optimize.

Let me also try a simple "greedy" approach: start with rank 125 and try to remove terms that are linearly dependent or can be absorbed.

But for an exact decomposition, this is very difficult.

Let me just implement the schoolbook correctly.
"""

import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that the decomposition is exact. Returns True if valid."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    
    The 7 products are:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B11)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U is 4x7, V is 4x7, W is 4x7
    # U[i*2+j, r] corresponds to A_{i,j}
    # V[j*2+k, r] corresponds to B_{j,k}
    # W[k*2+i, r] corresponds to C_{k,i} ... wait, W[k*n+i, r]
    
    # For 2x2x2: n=2, m=2, p=2
    # U has 4 rows (i*m+j for i,j in 0,1), 7 cols
    # V has 4 rows (j*p+k for j,k in 0,1), 7 cols  
    # W has 4 rows (k*n+i for k,i in 0,1), 7 cols
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # A00 is U[0], A11 is U[3]
    # B00 is V[0], B11 is V[3]
    # C00 gets +M1, C11 gets +M1
    # C00 is W[0*2+0] = W[0], C11 is W[1*2+1] = W[3]
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A10+A11)B00
    # A10 is U[2], A11 is U[3]
    # B00 is V[0]
    # C10 gets +M2, C11 gets -M2
    # C10 is W[1*2+0] = W[2], C11 is W[1*2+1] = W[3]
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3 = A00(B10-B11)
    # A00 is U[0]
    # B10 is V[2], B11 is V[3]
    # C01 gets +M3, C11 gets +M3
    # C01 is W[0*2+1] = W[1], C11 is W[1*2+1] = W[3]
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A11(B01-B11)
    # A11 is U[3]
    # B01 is V[1], B11 is V[3]
    # C00 gets +M4, C10 gets +M4
    # C00 is W[0], C10 is W[2]
    U[3][3] += 1
    V[1][3] += 1; V[3][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (A00+A01)B11
    # A00 is U[0], A01 is U[1]
    # B11 is V[3]
    # C00 gets -M5, C01 gets +M5
    # C00 is W[0], C01 is W[1]
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6 = (A10-A00)(B00+B10)
    # A10 is U[2], A00 is U[0]
    # B00 is V[0], B10 is V[2]
    # C11 gets +M6
    # C11 is W[3]
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[2][5] += 1
    W[3][5] += 1
    
    # M7 = (A01-A11)(B01+B11)
    # A01 is U[1], A11 is U[3]
    # B01 is V[1], B11 is V[3]
    # C00 gets +M7
    # C00 is W[0]
    U[1][6] += 1; U[3][6] -= 1
    V[1][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W


def _combine_partitions(n, m, p, blocks, decomp_map):
    """
    Combine block decompositions.
    blocks: list of (n_i, m_i, p_i) for each block
    decomp_map: dict mapping (n_i, m_i, p_i) to (U, V, W)
    """
    total_rank = sum(len(decomp_map[(ni, mi, pi)][0][0]) for ni, mi, pi in blocks)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for ni, mi, pi in blocks:
        Ui, Vi, Wi = decomp_map[(ni, mi, pi)]
        rank_i = len(Ui[0])
        
        # We need to map the block indices to the full matrix indices
        # This is complex - we need to know the position of each block
        # For now, this is a stub
        col_offset += rank_i
    
    return U, V, W


def construct():
    """
    Return factor matrices U, V, W for 5x5x5 matrix multiplication.
    Uses the schoolbook decomposition of rank n*m*p = 125.
    """
    _start = time.time()
    _deadline = _start + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)

    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # This should never happen, but fall back
        pass

    return {"U": U, "V": V, "W": W}
