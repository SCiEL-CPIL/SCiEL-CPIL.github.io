"""Tensor decomposition for 4x4x6 matrix multiplication using block Strassen (rank 77)."""

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 77
    U = [[0] * rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * rank for _ in range(p * n)]  # 24 x 77
    
    col = 0
    
    # --- PART 1: 4x4 * 4x4 -> 4x4 (columns 0..3 of B, rows 0..3 of C) ---
    # Partition into 2x2 blocks. For each (i0, j0, k0) in {0,1}^3:
    #   A block: rows 2*i0..2*i0+1, cols 2*j0..2*j0+1
    #   B block: rows 2*j0..2*j0+1, cols 2*k0..2*k0+1
    #   C block: rows 2*k0..2*k0+1, cols 2*i0..2*i0+1
    # Strassen 2x2x2 rank-7 terms per block.
    
    def add_strassen_block(i0, j0, k0, offset=0):
        """Add 7 rank-1 terms for Strassen 2x2x2 at block (i0,j0,k0).
        Returns next column index."""
        nonlocal col
        a00 = (2*i0) * m + (2*j0)      # A[2*i0, 2*j0]
        a01 = (2*i0) * m + (2*j0 + 1)  # A[2*i0, 2*j0+1]
        a10 = (2*i0+1) * m + (2*j0)    # A[2*i0+1, 2*j0]
        a11 = (2*i0+1) * m + (2*j0+1)  # A[2*i0+1, 2*j0+1]
        
        b00 = (2*j0) * p + (2*k0)      # B[2*j0, 2*k0]
        b01 = (2*j0) * p + (2*k0 + 1)  # B[2*j0, 2*k0+1]
        b10 = (2*j0+1) * p + (2*k0)    # B[2*j0+1, 2*k0]
        b11 = (2*j0+1) * p + (2*k0+1)  # B[2*j0+1, 2*k0+1]
        
        c00 = (2*k0) * n + (2*i0)      # C[2*k0, 2*i0]
        c01 = (2*k0) * n + (2*i0 + 1)  # C[2*k0, 2*i0+1]
        c10 = (2*k0+1) * n + (2*i0)    # C[2*k0+1, 2*i0]
        c11 = (2*k0+1) * n + (2*i0+1)  # C[2*k0+1, 2*i0+1]
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        # We split into 2 rank-1 terms:
        c = col; col += 1
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        c = col; col += 1
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c11][c] = 1
        
        # m2 = (c+d)e -> C10 += 1
        c = col; col += 1
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1
        
        # m3 = a(f-h) -> C01 += 1, C11 += 1
        c = col; col += 1
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c01][c] = 1
        c = col; col += 1
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c11][c] = 1
        
        # m4 = d(g-e) -> C00 += 1, C10 += 1
        c = col; col += 1
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c00][c] = 1
        c = col; col += 1
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c10][c] = 1
        
        # m5 = (a+b)h -> C00 -= 1, C01 += 1
        c = col; col += 1
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1
        c = col; col += 1
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c01][c] = 1
        
        # m6 = (c-a)(e+g) -> C11 += 1
        c = col; col += 1
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b10][c] = 1
        W[c11][c] = 1
        
        # m7 = (b-d)(f+h) -> C00 += 1
        c = col; col += 1
        U[a01][c] = 1; U[a11][c] = -1
        V[b01][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        
        return col
    
    # Add all 8 blocks for 4x4x4
    for i0 in range(2):
        for j0 in range(2):
            for k0 in range(2):
                add_strassen_block(i0, j0, k0)
    
    # --- PART 2: 4x4 * 4x2 -> 4x2 (columns 4..5 of B, rows 4..5 of C) ---
    # Partition A into 2x2 blocks (i0, j0), B rows into 2x2 blocks (j0), C rows into 2x2 blocks (i0).
    # B cols are 4,5 (no k-block splitting needed, just 2 columns).
    # For each (i0, j0) in {0,1}^2:
    #   A block: rows 2*i0..2*i0+1, cols 2*j0..2*j0+1
    #   B block: rows 2*j0..2*j0+1, cols 4..5
    #   C block: rows 4..5, cols 2*i0..2*i0+1
    # This is a 2x2 * 2x2 -> 2x2 problem (rank 7 each), 4 blocks total = 28 terms.
    
    def add_strassen_block_4x4x2(i0, j0):
        """Add 7 rank-1 terms for 2x2*2x2->2x2 at A-block(i0,j0), B-rows(2*j0..2*j0+1, cols 4-5), C-rows(4-5, cols 2*i0..2*i0+1)."""
        nonlocal col
        a00 = (2*i0) * m + (2*j0)
        a01 = (2*i0) * m + (2*j0 + 1)
        a10 = (2*i0+1) * m + (2*j0)
        a11 = (2*i0+1) * m + (2*j0+1)
        
        b00 = (2*j0) * p + 4
        b01 = (2*j0) * p + 5
        b10 = (2*j0+1) * p + 4
        b11 = (2*j0+1) * p + 5
        
        c00 = 4 * n + (2*i0)
        c01 = 4 * n + (2*i0 + 1)
        c10 = 5 * n + (2*i0)
        c11 = 5 * n + (2*i0 + 1)
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        c = col; col += 1
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        c = col; col += 1
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c11][c] = 1
        
        # m2 = (c+d)e -> C10 += 1
        c = col; col += 1
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1
        
        # m3 = a(f-h) -> C01 += 1, C11 += 1
        c = col; col += 1
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c01][c] = 1
        c = col; col += 1
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c11][c] = 1
        
        # m4 = d(g-e) -> C00 += 1, C10 += 1
        c = col; col += 1
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c00][c] = 1
        c = col; col += 1
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c10][c] = 1
        
        # m5 = (a+b)h -> C00 -= 1, C01 += 1
        c = col; col += 1
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1
        c = col; col += 1
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c01][c] = 1
        
        # m6 = (c-a)(e+g) -> C11 += 1
        c = col; col += 1
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b10][c] = 1
        W[c11][c] = 1
        
        # m7 = (b-d)(f+h) -> C00 += 1
        c = col; col += 1
        U[a01][c] = 1; U[a11][c] = -1
        V[b01][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        
        return col
    
    for i0 in range(2):
        for j0 in range(2):
            add_strassen_block_4x4x2(i0, j0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
