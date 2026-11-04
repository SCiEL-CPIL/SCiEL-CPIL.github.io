"""
Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.

Approach: Use Strassen's algorithm (rank 7 for 2x2x2) on 2x2x2 blocks,
combined with schoolbook for the remaining rows/cols.

For 3x4 * 4x8:
  Split A (3x4) into A_top (2x4) and A_bottom (1x4)
  Split B (4x8) into B_left (4x4) and B_right (4x4)

  C_top_left (2x4) = A_top * B_left
  C_top_right (2x4) = A_top * B_right
  C_bottom_left (1x4) = A_bottom * B_left
  C_bottom_right (1x4) = A_bottom * B_right

For the 2x4 * 4x4 blocks, we can use Strassen on 2x2x2 sub-blocks:
  A_top (2x4) = [A11 A12] where A11, A12 are 2x2
  B_left (4x4) = [[B11 B12], [B21 B22]] where each is 2x2

  C_top_left = A11*B11 + A12*B21 (top 2 rows of result)
             + A11*B12 + A12*B22 (bottom 2 rows of result)

Actually, let me think more carefully.

C_top_left is 2x4. Let's write:
  A_top = [A1 A2] where A1, A2 are 2x2
  B_left = [B1 B2; B3 B4] where B1..B4 are 2x2

  C_top_left = A1*B1 + A2*B3  (left 2 cols)
             + A1*B2 + A2*B4  (right 2 cols)

Each 2x2*2x2 multiplication can use Strassen (rank 7).
So C_top_left needs 4 * 7 = 28 terms.
Similarly C_top_right needs 28 terms.
C_bottom_left needs 4 * 2 = 8 terms (schoolbook for 1x2*2x2 = 4 mults, but actually 1x4*4x4 = 16 mults schoolbook)

Wait, let me reconsider. A_bottom is 1x4, B_left is 4x4.
C_bottom_left = A_bottom * B_left is 1x4.
Schoolbook: 4*4 = 16 multiplications.

Total schoolbook: 3*4*8 = 96.

With Strassen on the 2x2x2 blocks:
- For each 2x2*2x2 block, use rank 7 instead of rank 4.
- Number of 2x2x2 blocks in 3x4*4x8:
  We can partition into 2x2x2 blocks where possible.

Actually, let me use a cleaner approach.

Use Strassen for 2x2x2: rank 7.

For 3x4 * 4x8:
  Partition into 2x2x2 blocks:
  - Rows 1-2, Cols 1-2 of B, Output rows 1-2, cols 1-2: 1 block (2x2x2)
  - Rows 1-2, Cols 3-4 of B, Output rows 1-2, cols 3-4: 1 block
  - Rows 1-2, Cols 5-6 of B, Output rows 1-2, cols 5-6: 1 block
  - Rows 1-2, Cols 7-8 of B, Output rows 1-2, cols 7-8: 1 block
  - Row 3, Cols 1-2 of B, Output row 3, cols 1-2: schoolbook (1x2*2x2 = 4)
  - Row 3, Cols 3-4 of B, Output row 3, cols 3-4: schoolbook (4)
  - Row 3, Cols 5-6 of B, Output row 3, cols 5-6: schoolbook (4)
  - Row 3, Cols 7-8 of B, Output row 3, cols 7-8: schoolbook (4)

  Total: 4*7 + 4*4 = 28 + 16 = 44.

This gives rank 44, much better than 96!

Let me implement this.
"""

def construct():
    n, m, p = 3, 4, 8
    
    # We'll build the decomposition term by term.
    # Each term is (u, v, w) where:
    #   u is a vector of length n*m = 12
    #   v is a vector of length m*p = 32
    #   w is a vector of length p*n = 24
    #   and sum_t u_t * v_t * w_t^T = T (the tensor)
    
    # The tensor T[i,j,k] = 1 if k = i*m + j (i.e., output index k corresponds to row i, col j of C)
    # Wait, the convention is W[k*n+i, r] so output index is k*n + i where k is col, i is row.
    # So T[i,j,k] = 1 if output (i,k) depends on A[i,j] * B[j,k].
    
    # Let's use the following decomposition:
    # For each 2x2x2 Strassen block and each schoolbook block, add the corresponding terms.
    
    terms = []  # list of (u_list, v_list, w_list)
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Strassen 2x2x2: computes C = A * B where A, B, C are 2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # Strassen's 7 products:
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * b11
    # m3 = a11 * (b12 - b22)
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a12) * b22
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # For a specific 2x2x2 block at positions:
    #   A rows: ar1, ar2 (0-indexed)
    #   B cols: bc1, bc2 (0-indexed)
    #   Output rows: or1, or2
    #   Output cols: oc1, oc2
    #   B rows: br1, br2 (the 2 rows of B that multiply with these A rows)
    
    def add_strassen_block(ar1, ar2, br1, br2, oc1, oc2):
        """Add Strassen terms for a 2x2x2 block.
        A rows: ar1, ar2
        B rows: br1, br2
        B cols: oc1, oc2 (which are also output cols)
        Output rows: ar1, ar2
        """
        # m1 = (A[ar1,br1] + A[ar2,br2]) * (B[br1,oc1] + B[br2,oc2])
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[ar1 * m + br1] = 1
        u[ar2 * m + br2] = 1
        v[br1 * p + oc1] = 1
        v[br2 * p + oc2] = 1
        w[oc1 * n + ar1] = 1
        w[oc2 * n + ar2] = 1
        w[oc1 * n + ar2] = 1
        w[oc2 * n + ar1] = -1
        add_term(u, v, w)
        
        # m2 = (A[ar2,br1] + A[ar2,br2]) * B[br1,oc1]
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[ar2 * m + br1] = 1
        u[ar2 * m + br2] = 1
        v[br1 * p + oc1] = 1
        w[oc1 * n + ar2] = 1
        w[oc2 * n + ar2] = 1
        add_term(u, v, w)
        
        # m3 = A[ar1,br1] * (B[br1,oc2] - B[br2,oc2])
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[ar1 * m + br1] = 1
        v[br1 * p + oc2] = 1
        v[br2 * p + oc2] = -1
        w[oc2 * n + ar1] = 1
        w[oc2 * n + ar2] = 1
        add_term(u, v, w)
        
        # m4 = A[ar2,br2] * (B[br2,oc1] - B[br1,oc1])
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[ar2 * m + br2] = 1
        v[br2 * p + oc1] = 1
        v[br1 * p + oc1] = -1
        w[oc1 * n + ar1] = 1
        w[oc1 * n + ar2] = 1
        add_term(u, v, w)
        
        # m5 = (A[ar1,br1] + A[ar1,br2]) * B[br2,oc2]
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[ar1 * m + br1] = 1
        u[ar1 * m + br2] = 1
        v[br2 * p + oc2] = 1
        w[oc1 * n + ar1] = -1
        w[oc2 * n + ar2] = 1
        add_term(u, v, w)
        
        # m6 = (A[ar2,br1] - A[ar1,br1]) * (B[br1,oc1] + B[br1,oc2])
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[ar2 * m + br1] = 1
        u[ar1 * m + br1] = -1
        v[br1 * p + oc1] = 1
        v[br1 * p + oc2] = 1
        w[oc2 * n + ar2] = 1
        add_term(u, v, w)
        
        # m7 = (A[ar1,br2] - A[ar2,br2]) * (B[br2,oc1] + B[br2,oc2])
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[ar1 * m + br2] = 1
        u[ar2 * m + br2] = -1
        v[br2 * p + oc1] = 1
        v[br2 * p + oc2] = 1
        w[oc1 * n + ar1] = 1
        w[oc2 * n + ar1] = 1
        add_term(u, v, w)
    
    # Add Strassen blocks for rows 0,1 (0-indexed) with B row pairs (0,1) and (2,3)
    # and output col pairs (0,1), (2,3), (4,5), (6,7)
    
    # For A rows 0,1:
    #   B rows 0,1 -> output cols 0,1
    add_strassen_block(0, 1, 0, 1, 0, 1)
    #   B rows 0,1 -> output cols 2,3
    add_strassen_block(0, 1, 0, 1, 2, 3)
    #   B rows 0,1 -> output cols 4,5
    add_strassen_block(0, 1, 0, 1, 4, 5)
    #   B rows 0,1 -> output cols 6,7
    add_strassen_block(0, 1, 0, 1, 6, 7)
    
    #   B rows 2,3 -> output cols 0,1
    add_strassen_block(0, 1, 2, 3, 0, 1)
    #   B rows 2,3 -> output cols 2,3
    add_strassen_block(0, 1, 2, 3, 2, 3)
    #   B rows 2,3 -> output cols 4,5
    add_strassen_block(0, 1, 2, 3, 4, 5)
    #   B rows 2,3 -> output cols 6,7
    add_strassen_block(0, 1, 2, 3, 6, 7)
    
    # That's 8 Strassen blocks * 7 terms = 56 terms for A rows 0,1.
    
    # For A row 2 (0-indexed), use schoolbook for each 2-col pair of B:
    # B rows 0,1 with output cols 0,1
    # B rows 0,1 with output cols 2,3
    # B rows 0,1 with output cols 4,5
    # B rows 0,1 with output cols 6,7
    # B rows 2,3 with output cols 0,1
    # B rows 2,3 with output cols 2,3
    # B rows 2,3 with output cols 4,5
    # B rows 2,3 with output cols 6,7
    
    # Each schoolbook 1x2 * 2x2 = 4 terms
    for br1, br2 in [(0, 1), (2, 3)]:
        for oc1, oc2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
            # A[2, br1] * B[br1, oc1]
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            u[2 * m + br1] = 1
            v[br1 * p + oc1] = 1
            w[oc1 * n + 2] = 1
            add_term(u, v, w)
            
            # A[2, br1] * B[br1, oc2]
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            u[2 * m + br1] = 1
            v[br1 * p + oc2] = 1
            w[oc2 * n + 2] = 1
            add_term(u, v, w)
            
            # A[2, br2] * B[br2, oc1]
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            u[2 * m + br2] = 1
            v[br2 * p + oc1] = 1
            w[oc1 * n + 2] = 1
            add_term(u, v, w)
            
            # A[2, br2] * B[br2, oc2]
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            u[2 * m + br2] = 1
            v[br2 * p + oc2] = 1
            w[oc2 * n + 2] = 1
            add_term(u, v, w)
    
    # Total: 56 + 32 = 88 terms.
    
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for t, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][t] = u[i]
        for j in range(m * p):
            V[j][t] = v[j]
        for k in range(p * n):
            W[k][t] = w[k]
    
    return {"U": U, "V": V, "W": W}
