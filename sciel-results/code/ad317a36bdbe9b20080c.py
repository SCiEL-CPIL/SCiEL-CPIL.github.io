"""Different strategy: use a known rank-7 decomposition for 3x2 * 2x2 blocks via embedding into 3x3 Strassen-like structure, combined with careful block partitioning.

The current program uses 2x2 Strassen for i={0,1} and schoolbook for i={2}, giving rank 88.
A different structural approach: partition j into two 2-dim blocks, k into four 2-dim blocks,
and for each (j-block, k-block) pair, use a rank-7 decomposition for the 3x2 * 2x2 sub-multiplication.

Key insight: A 3x2 * 2x2 multiplication can be decomposed with rank 7 by treating it as part of a 3x3 * 3x3 Strassen decomposition where the third row/column is zero. This is a different algebraic structure than the current 2x2 Strassen + 1x1 schoolbook split.

Actually, let's think more carefully. The 3x2 * 2x2 tensor has dimensions (3*2, 2*2) = (6, 4) in U, (4, 4) in V... wait no.

For a 3x2 * 2x2 block:
- U has shape (3*2, rank) = (6, rank)
- V has shape (2*2, rank) = (4, rank)  
- W has shape (2*3, rank) = (6, rank)

The tensor T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1}, k in {0,1}.

This is a 3x2x2 tensor. The schoolbook rank is 3*2*2 = 12.
Can we do better? 

A 3x2x2 tensor: C[i,k] = sum_j A[i,j] B[j,k] for i in 0..2, k in 0..1.
This is equivalent to multiplying a 3x2 matrix by a 2x2 matrix.

We can embed this into a 3x3 * 3x3 multiplication by adding a zero row and column.
The 3x3 Strassen rank is 23, but that's for the full 3x3*3x3. For our embedded case, we might be able to use a subset.

Actually, let's try a different approach entirely: use the fact that 3x2*2x2 can be done with rank 8 or 9 using a modified Strassen.

Let me just implement a clean block-diagonal decomposition with 3x2*2x2 blocks and see if we can find a rank-7 or rank-8 decomposition for each block.

For now, I'll implement the block structure and use schoolbook (rank 12) per block as a baseline, then try to improve.

Wait - the current program achieves 88. Let me count: 2 j-blocks * 4 k-blocks * (7 + 4) = 88.

If I use 3x2*2x2 blocks: 2 j-blocks * 4 k-blocks * rank_per_block.
If rank_per_block = 11, total = 88. Same.
If rank_per_block = 10, total = 80. Better!
If rank_per_block = 9, total = 72. Much better!

So the key question: what is the rank of the 3x2x2 matrix multiplication tensor?

The 3x2x2 tensor T_{ijk} where C_{ik} = sum_j A_{ij} B_{jk}.
Matricizations:
- (3*2) x 2: rank 2 (since C has 2 columns)
- 2 x (2*2): rank 2
- 2 x (3*2): rank 2

The border rank is at least 2, but the actual rank could be higher.

For 1x2*2x1 (vector-matrix-vector), rank is 2.
For 2x2*2x2, rank is 7 (Strassen).
For 3x2*2x2, I believe the rank is 10 or 11.

Let me just implement the block structure and use a reasonable decomposition.
"""

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_3x2x2_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """
    Decompose a 3x2 * 2x2 block using a rank-10 decomposition.
    
    For a 3x2 * 2x2 multiplication:
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] for i in 0..2, k in 0..1
    
    We can use the following approach:
    - Use Strassen-like formulas for the top 2x2 part (rows 0,1)
    - Handle row 2 separately or integrate it
    
    Actually, let's use a simpler approach: treat it as two separate multiplications
    and try to share some terms.
    
    A rank-10 decomposition for 3x2x2:
    We can think of this as:
    - Rows 0,1: 2x2 * 2x2, rank 7 (Strassen)
    - Row 2: 1x2 * 2x2, rank 4 (schoolbook)
    Total: 11
    
    But we can potentially share terms between the Strassen part and the row-2 part.
    
    Let me just use rank 11 (7 + 4) for now, which gives total 88.
    To beat 88, I need to find a better decomposition.
    
    Alternative: Use a rank-9 decomposition for 3x2x2.
    """
    # Use Strassen for rows 0,1 (2x2 * 2x2)
    col = col_start
    terms_strassen = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    for t in range(7):
        a_coeffs = terms_strassen[t][0]
        b_coeffs = terms_strassen[t][1]
        c_coeffs = terms_strassen[t][2]
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
        for dj in range(2):
            for dk in range(2):
                V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
        for di in range(2):
            for dk in range(2):
                W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
        col += 1
    
    # Schoolbook for row 2 (1x2 * 2x2)
    for dj in range(2):
        for dk in range(2):
            U[i0*2*m + 2*m + (j0+dj)][col] = 1  # This is wrong indexing
            col += 1
    
    # The above indexing is wrong. Let me fix it.
    # For the 3x2x2 block starting at i0, j0, k0:
    # U indices: (i0+0)*m + j0, (i0+1)*m + j0, (i0+2)*m + j0 for the three i values
    # But m=4, so the actual global indices are different.
    
    # Let me redo this with proper indexing.
    col = col_start
    
    # Strassen for i={i0, i0+1}, j={j0, j0+1}, k={k0, k0+1}
    terms_strassen = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    for t in range(7):
        a_coeffs = terms_strassen[t][0]
        b_coeffs = terms_strassen[t][1]
        c_coeffs = terms_strassen[t][2]
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
        for dj in range(2):
            for dk in range(2):
                V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
        for di in range(2):
            for dk in range(2):
                W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
        col += 1
    
    # Schoolbook for i={i0+2}, j={j0, j0+1}, k={k0, k0+1}
    for dj in range(2):
        for dk in range(2):
            U[(i0+2)*m + (j0+dj)][col] = 1
            V[(j0+dj)*p + (k0+dk)][col] = 1
            W[(k0+dk)*n + (i0+2)][col] = 1
            col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Block structure: j into 2 blocks of 2, k into 4 blocks of 2
    # For each (j-block, k-block), use a 3x2x2 decomposition
    j_blocks = [0, 2]
    k_blocks = [0, 2, 4, 6]
    
    # Each 3x2x2 block uses rank 11 (7 Strassen + 4 schoolbook)
    rank_per_block = 11
    total_blocks = len(j_blocks) * len(k_blocks)
    rank = rank_per_block * total_blocks  # 11 * 8 = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in j_blocks:
        for k0 in k_blocks:
            col = _strassen_3x2x2_block(U, V, W, 0, j0, k0, col, n, m, p)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
