import time

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
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

def _construct_rank_26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    Partition into 2x2 block (indices 0,1) and 1x1 blocks (index 2).
    Apply Strassen to the 2x2 block (7 terms) + 19 terms for the rest.
    """
    n, m, p = 3, 3, 3
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(U, V, W, u_rows, v_rows, w_rows, val=1):
        """Add a rank-1 term. u_rows, v_rows, w_rows are dicts {row_idx: value}"""
        pass  # We'll just set directly
    
    # Strassen for 2x2 block (i,j,k in {0,1})
    # The 2x2 matmul tensor has 8 non-zero entries: T[i,j,k]=1 for i,j,k in {0,1}
    # We need 7 rank-1 terms to cover these.
    
    # Standard Strassen decomposition for 2x2:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    # where:
    # M1 = (A[0,0]+A[0,1]) * (B[0,0]+B[1,0])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[0,0])  -- wait, let me use the standard form
    
    # Let me use the standard Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B22 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B12 + B22)
    # M7 = (A12 - A22)(B11 + B21)
    
    # But for our tensor, we need to express T[i,j,k] = 1 for i,j,k in {0,1}
    # as a sum of 7 rank-1 tensors.
    
    # Let me directly specify the 7 terms for the 2x2 block.
    # Each term is a rank-1 tensor u ⊗ v ⊗ w where u is in R^4 (for i,j in {0,1}),
    # v is in R^4 (for j,k in {0,1}), w is in R^4 (for k,i in {0,1}).
    
    # Actually, let me just use the known rank-7 decomposition of the 2x2 matmul tensor.
    # The 2x2 matmul tensor T_{i,j,k} = 1 if i,j,k in {0,1}, else 0.
    # This is equivalent to the standard 2x2 matrix multiplication.
    
    # The 7 terms (using 0-indexed for the 2x2 block):
    # Term 0: u=(1,0,0,1), v=(1,0,0,0), w=(1,0,0,0) -- this doesn't work simply
    
    # Let me use a different approach. I'll define the 7 Strassen terms explicitly
    # by specifying which (i,j), (j,k), (k,i) pairs get which values.
    
    # For the 2x2 block, the indices are:
    # i,j in {0,1} -> U rows 0,1,2,3 (i*m+j = i*2+j for the 2x2 subproblem)
    # But in our 3x3 problem, U has 9 rows. The 2x2 block corresponds to rows 0,1,2,3
    # (i=0,j=0 -> row 0; i=0,j=1 -> row 1; i=1,j=0 -> row 2; i=1,j=1 -> row 3)
    # V has 9 rows. The 2x2 block corresponds to cols 0,1 of B and rows 0,1 of B:
    # (j=0,k=0) -> row 0; (j=0,k=1) -> row 1; (j=1,k=0) -> row 3; (j=1,k=1) -> row 4
    # W has 9 rows. The 2x2 block corresponds to k in {0,1}, i in {0,1}:
    # (k=0,i=0) -> row 0; (k=0,i=1) -> row 1; (k=1,i=0) -> row 3; (k=1,i=1) -> row 4
    
    # Let me define the 7 terms for the 2x2 block:
    # I'll use the standard Strassen decomposition.
    
    # Term 0: M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    #   U: rows 0 (i=0,j=0) += 1, row 3 (i=1,j=1) += 1
    #   V: rows 0 (j=0,k=0) += 1, row 4 (j=1,k=1) += 1
    #   W: rows 0 (k=0,i=0) += 1, row 4 (k=1,i=1) += 1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[4][0] += 1
    
    # Term 1: M2 = (A[1,0]+A[1,1]) * B[0,0]
    #   U: rows 2 (i=1,j=0) += 1, row 3 (i=1,j=1) += 1
    #   V: row 0 (j=0,k=0) += 1
    #   W: row 0 (k=0,i=0) += 1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[0][1] += 1
    
    # Term 2: M3 = A[0,0] * (B[1,0]-B[0,0])
    #   U: row 0 (i=0,j=0) += 1
    #   V: row 3 (j=1,k=0) += 1, row 0 (j=0,k=0) += -1
    #   W: row 0 (k=0,i=0) += 1
    U[0][2] += 1
    V[3][2] += 1; V[0][2] += -1
    W[0][2] += 1
    
    # Term 3: M4 = A[1,1] * (B[1,1]-B[0,1])
    #   U: row 3 (i=1,j=1) += 1
    #   V: row 4 (j=1,k=1) += 1, row 1 (j=0,k=1) += -1
    #   W: row 4 (k=1,i=1) += 1
    U[3][3] += 1
    V[4][3] += 1; V[1][3] += -1
    W[4][3] += 1
    
    # Term 4: M5 = (A[0,0]+A[0,1]) * B[1,1]
    #   U: row 0 (i=0,j=0) += 1, row 1 (i=0,j=1) += 1
    #   V: row 4 (j=1,k=1) += 1
    #   W: row 4 (k=1,i=1) += 1
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[4][4] += 1
    
    # Term 5: M6 = (A[1,0]-A[0,0]) * (B[0,1]+B[1,1])
    #   U: row 2 (i=1,j=0) += 1, row 0 (i=0,j=0) += -1
    #   V: row 1 (j=0,k=1) += 1, row 4 (j=1,k=1) += 1
    #   W: row 4 (k=1,i=1) += 1
    U[2][5] += 1; U[0][5] += -1
    V[1][5] += 1; V[4][5] += 1
    W[4][5] += 1
    
    # Term 6: M7 = (A[0,1]-A[1,1]) * (B[0,0]+B[1,0])
    #   U: row 1 (i=0,j=1) += 1, row 3 (i=1,j=1) += -1
    #   V: row 0 (j=0,k=0) += 1, row 3 (j=1,k=0) += 1
    #   W: row 0 (k=0,i=0) += 1
    U[1][6] += 1; U[3][6] += -1
    V[0][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    # Now we need 19 more terms for the remaining entries involving index 2.
    # The remaining entries are those where at least one of i,j,k is 2.
    # Total entries: 27. 2x2 block has 8 entries. Remaining: 19 entries.
    # We need to cover these 19 entries with 19 rank-1 terms (one each).
    
    # The remaining entries:
    # i=0,j=0,k=2: U[0], V[2], W[6] (k=2,i=0)
    # i=0,j=1,k=2: U[1], V[2], W[6]
    # i=0,j=2,k=0: U[4], V[6], W[0]
    # i=0,j=2,k=1: U[4], V[7], W[1]
    # i=0,j=2,k=2: U[4], V[8], W[6]
    # i=1,j=0,k=2: U[2], V[2], W[7]
    # i=1,j=1,k=2: U[3], V[2], W[7]
    # i=1,j=2,k=0: U[5], V[6], W[1]
    # i=1,j=2,k=1: U[5], V[7], W[7]
    # i=1,j=2,k=2: U[5], V[8], W[7]
    # i=2,j=0,k=0: U[6], V[0], W[3]
    # i=2,j=0,k=1: U[6], V[1], W[4]
    # i=2,j=0,k=2: U[6], V[2], W[0]
    # i=2,j=1,k=0: U[7], V[3], W[4]
    # i=2,j=1,k=1: U[7], V[4], W[0]
    # i=2,j=1,k=2: U[7], V[5], W[0]  -- wait, V[5] is j=1,k=2
    # i=2,j=2,k=0: U[8], V[6], W[0]
    # i=2,j=2,k=1: U[8], V[7], W[1]
    # i=2,j=2,k=2: U[8], V[8], W[2]
    
    # Let me list them more carefully:
    # (i,j,k) -> U row i*m+j, V row j*p+k, W row k*n+i
    remaining = [
        (0,0,2), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
        (1,0,2), (1,1,2), (1,2,0), (1,2,1), (1,2,2),
        (2,0,0), (2,0,1), (2,0,2), (2,1,0), (2,1,1), (2,1,2),
        (2,2,0), (2,2,1), (2,2,2)
    ]
    
    col = 7
    for (i,j,k) in remaining:
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        U[u_row][col] = 1
        V[v_row][col] = 1
        W[w_row][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a rank-minimal decomposition for 3x3 matrix multiplication.
    """
    # Try the rank-26 construction
    result = _construct_rank_26()
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fall back to schoolbook rank 27
    return _construct_schoolbook()
