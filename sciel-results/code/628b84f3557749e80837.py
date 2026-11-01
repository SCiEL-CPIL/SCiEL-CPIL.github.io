import time

def construct():
    """
    Rank-42 exact decomposition for 2x4 @ 4x6 matrix multiplication.
    
    The tensor rank for 2x4x6 matrix multiplication is 42.
    This is achieved by splitting the 4x4 middle dimension into two 2x2 blocks.
    C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    Each term is a 2x2 @ 2x6 multiplication.
    A 2x2 @ 2x6 multiplication can be decomposed into three 2x2 @ 2x2 multiplications.
    Each 2x2 @ 2x2 multiplication has rank 7 (Strassen).
    Total rank = 2 * (3 * 7) = 42.
    
    Interface:
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)
    W: (p*n, rank) = (12, 42)
    
    Indexing conventions:
    U index: i * m + j  (for A[i, j])
    V index: j * p + k  (for B[j, k])
    W index: k * n + i  (for C[i, k]) -- Note: Prompt says W[k*n+i, r]
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2x2 core: 7 multiplications
    # For X @ Y where X, Y are 2x2:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Result:
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # Define Strassen pattern for 2x2 @ 2x2
    # Each term is a dict with 'A', 'B', 'W' lists of (row, col, sign)
    # A, B are 2x2 sub-matrices. W is 2x2 sub-matrix of output.
    
    strassen_terms = [
        # M1: (A00+A11)*(B00+B11) -> Z00(+1), Z11(+1)
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2: (A10+A11)*B00 -> Z10(+1), Z11(-1)
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'W': [(1,0,1), (1,1,-1)]
        },
        # M3: A00*(B01-B11) -> Z01(+1), Z11(+1)
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'W': [(0,1,1), (1,1,1)]
        },
        # M4: A11*(B10-B00) -> Z00(+1), Z10(+1)
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (1,0,1)]
        },
        # M5: (A00+A01)*B11 -> Z00(-1), Z01(+1)
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'W': [(0,0,-1), (0,1,1)]
        },
        # M6: (A10-A00)*(B00+B01) -> Z11(+1)
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7: (A01-A11)*(B10+B11) -> Z00(+1)
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    col = 0
    
    # Helper: apply Strassen for a 2x2 @ 2x2 sub-problem
    # a_base: (row_offset, col_offset) in A (2x4 matrix)
    # b_base: (row_offset, col_offset) in B (4x6 matrix)
    # out_base: (row_offset, col_offset) in C (2x6 matrix)
    def apply_strassen(a_base, b_base, out_base):
        nonlocal col
        for term in strassen_terms:
            # Set U coefficients
            for (ar, ac, sign) in term['A']:
                # A element at (a_base[0]+ar, a_base[1]+ac)
                # U index: (row) * m + (col)
                u_idx = (a_base[0] + ar) * m + (a_base[1] + ac)
                U[u_idx][col] += sign
            
            # Set V coefficients
            for (br, bc, sign) in term['B']:
                # B element at (b_base[0]+br, b_base[1]+bc)
                # V index: (row) * p + (col)
                v_idx = (b_base[0] + br) * p + (b_base[1] + bc)
                V[v_idx][col] += sign
            
            # Set W coefficients
            for (or_, oc, sign) in term['W']:
                # Output element at (out_base[0]+or_, out_base[1]+oc) in C (2x6)
                # W index: col * n + row = (out_base[1]+oc) * 2 + (out_base[0]+or_)
                w_idx = (out_base[1] + oc) * n + (out_base[0] + or_)
                W[w_idx][col] += sign
            
            col += 1
    
    # Decompose 2x4 @ 4x6 into two 2x2 @ 2x6 blocks
    # Block 1: A[:, 0:2] @ B[0:2, :]
    # Block 2: A[:, 2:4] @ B[2:4, :]
    
    # For each 2x2 @ 2x6 block, split into three 2x2 @ 2x2 blocks
    # B cols 0-1, 2-3, 4-5
    
    # Block 1: Left half of A, Top half of B
    for sub in range(3):
        b_col_offset = 2 * sub
        out_col_offset = 2 * sub
        apply_strassen((0, 0), (0, b_col_offset), (0, out_col_offset))
        
    # Block 2: Right half of A, Bottom half of B
    for sub in range(3):
        b_col_offset = 2 * sub
        out_col_offset = 2 * sub
        apply_strassen((0, 2), (2, b_col_offset), (0, out_col_offset))
    
    # Verify rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
