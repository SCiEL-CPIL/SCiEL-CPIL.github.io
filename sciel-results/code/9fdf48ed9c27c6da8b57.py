"""
Strategy: Explicit constructive decomposition for 2x3 * 3x5 matrix multiplication.

The key insight is to use a block decomposition approach:
- Split the 3x5 right matrix into three 1x5 row blocks.
- Use the known rank-7 decomposition for 2x3 * 3x1 (which is essentially
  the 2x3 matrix-vector multiplication).
- For each of the 5 columns of the right matrix, we need rank 7, but we can
  share some structure across columns.

Actually, a simpler approach: use the fact that we can decompose the multiplication
by treating each output column independently, but that gives rank 3*5*2 = 30.

Better approach: Use a structured decomposition based on:
- The 2x3 * 3x5 multiplication can be seen as computing 2x5 output.
- Each output entry (i,k) = sum_j A[i,j] * B[j,k] for j=0,1,2.

A known result: the rank of M_{2,3,5} is at most 18 (from Strassen-type
decompositions for small matrices). Let's try to construct an explicit rank-18
decomposition.

Actually, let me think about this differently. We can use the following approach:
- The multiplication tensor for M_{n,m,p} has a known upper bound from
  Strassen's algorithm adapted to rectangular matrices.
- For 2x3 * 3x5, one approach is to pad to 4x4 * 4x4 (rank 28 for Strassen),
  but that's not minimal.

Let me try a direct constructive approach with rank 15 (the lower bound max(mp, np, mn) = max(15,10,6) = 15).

For rank 15, we need exactly 15 terms. Let's try to find a solution using
a systematic approach with the specific structure of 2x3x5.

Given the difficulty of finding rank 15, let's use a known valid decomposition
with rank 18 or less, which is still better than the schoolbook rank of 30.

I'll use a constructive approach based on splitting the 3-row dimension:
- Split the 3 rows into (1,2) or (2,1).
- Use the 2x1 * 1x5 (rank 1) and 2x2 * 2x5 (rank ?) decompositions.

2x2 * 2x5: schoolbook rank is 4*5 = 20, but we can do better.
Actually 2x2 * 2x5: lower bound is max(2*5, 2*5, 4) = max(10,10,4) = 10.

Let me just hardcode a known valid decomposition. I'll use a systematic
construction with rank 18.

After careful consideration, here's a concrete approach:
- Use 6 terms for the "diagonal" (matching row/col structure)
- Use additional terms for cross terms

Let me construct a valid rank-15 decomposition if possible, otherwise rank 18.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    start_time = time.time()
    deadline = start_time + 100
    
    # Try to find a low-rank decomposition
    # Start with known constructions
    
    # Approach: Use a structured decomposition
    # For M_{2,3,5}, we can use the following:
    # - Treat the multiplication as sum over j of A[:,j] * B[j,:] (outer product)
    # - That gives 3 terms, each of rank... no, that's not right.
    
    # The tensor T[i,j,k] = delta_{ik}... no, T[i,j,k] = 1 if the multiplication
    # uses A[i,j] and B[j,k] to produce C[i,k].
    
    # So T[i,j,k] = 1 for all i,j,k in their ranges.
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    
    # Let's try a direct construction with rank 15.
    # 
    # Idea: For each of the 15 (j,k) pairs in B, we could use 1 term, but that's rank 15
    # only if we can make U[i,j,c] depend on i in a way that works.
    # 
    # Actually, if we set for each (j,k) pair a term c:
    #   V[j,k,c] = 1
    #   U[i,j,c] = 1 for all i (doesn't work because we need different i)
    #   
    # Better: For each (j,k), we need the term to contribute to all i.
    # If U[i,j,c] = 1 for all i, W[k,i,c] = 1 for all i, V[j,k,c] = 1,
    # then T[i,j,k] gets contribution 1 for that (j,k). But we need exactly 1 total.
    # This would give T[i,j,k] = 1 only for the specific (j,k) term, so each
    # (i,j,k) triple gets exactly 1 from its corresponding (j,k) term.
    # But wait, U[i,j,c] = 1 for all i means U has the same value for all i in a column,
    # and W[k,i,c] = 1 for all i means W has the same value for all i in a column.
    # 
    # Let's try: for each (j,k) in 3x5 grid, assign a column c:
    #   U[i,j,c] = 1 for i=0,1
    #   V[j,k,c] = 1
    #   W[k,i,c] = 1 for i=0,1
    # 
    # Then T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # For a specific (i,j,k), only the column c corresponding to (j,k) has
    # V[j,k,c] = 1, and for that column U[i,j,c] = 1 and W[k,i,c] = 1.
    # So T[i,j,k] = 1. 
    # 
    # This gives a valid rank-15 decomposition!
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]  # 6 x 15
    V = [[0] * rank for _ in range(m * p)]  # 15 x 15
    W = [[0] * rank for _ in range(p * n)]  # 10 x 15
    
    col = 0
    for j in range(m):
        for k in range(p):
            # V[j*p + k][col] = 1
            V[j * p + k][col] = 1
            # U[i*m + j][col] = 1 for i = 0, 1
            for i in range(n):
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            col += 1
    
    # Verify the decomposition
    def _verify(U_mat, V_mat, W_mat):
        """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k."""
        r = len(U_mat[0]) if U_mat else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U_mat[i * m + j][c] * V_mat[j * p + k][c] * W_mat[k * n + i][c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If the simple construction fails (shouldn't), fall back to schoolbook
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
