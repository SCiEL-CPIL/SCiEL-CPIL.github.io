"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into m1=2, m2=3
    m1, m2 = 2, 3
    # Split p=6 into p1=2, p2=2, p3=2 for the m1 part (3 blocks of 2x2)
    # And p4=3, p5=3 for the m2 part (2 blocks of 3x3)
    # Actually, p is the output width.
    # Part 1: A[:, 0:2] (2x2) * B[0:2, :] (2x6)
    #   B[0:2, :] is split into 3 blocks of 2x2.
    #   Each 2x2 * 2x2 uses Strassen (Rank 7). Total 21.
    # Part 2: A[:, 2:5] (2x3) * B[2:5, :] (3x6)
    #   B[2:5, :] is split into 2 blocks of 3x3.
    #   Each 2x3 * 3x3 uses Schoolbook (Rank 18). Total 36.
    # Total Rank = 21 + 36 = 57.
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]  # 10 x 57
    V = [[0] * rank for _ in range(m * p)]  # 30 x 57
    W = [[0] * rank for _ in range(p * n)]  # 12 x 57
    
    col = 0
    
    # Helper to add Strassen 2x2x2 term
    def add_strassen(a_base, b_base, w_base, rank_offset):
        # a_base: row index in A (0 or 1)
        # b_base: col index in B (0, 2, 4)
        # w_base: col index in C (0, 2, 4)
        # Local indices for A: (0,0), (0,1), (1,0), (1,1)
        # Local indices for B: (0,0), (0,1), (1,0), (1,1)
        # Local indices for C: (0,0), (0,1), (1,0), (1,1)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a_base + 0][col + rank_offset] += 1
        U[a_base + 1][col + rank_offset] += 1
        V[b_base + 0][col + rank_offset] += 1
        V[b_base + 1][col + rank_offset] += 1
        W[w_base + 0][col + rank_offset] += 1
        W[w_base + 1][col + rank_offset] += 1
        
        # M2 = (A10 + A11)B00
        U[a_base + 1][col + rank_offset + 1] += 1
        U[a_base + 1][col + rank_offset + 1] += 1 # Wait, A11 is row 1, col 1. A10 is row 1, col 0.
        # Correction:
        U[a_base + 1][col + rank_offset + 1] = 1 # A10
        U[a_base + 1][col + rank_offset + 1] += 1 # A11 -> 2? No.
        
        # Let's do it cleanly.
        pass

    # I will implement the 57 rank decomposition explicitly without complex helpers to avoid bugs.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 3 Strassen blocks (2x2 * 2x2)
    # A is 2x2 (rows 0-1, cols 0-1)
    # B blocks are at cols 0-1, 2-3, 4-5
    # C blocks are at cols 0-1, 2-3, 4-5
    
    strassen_coeffs = [
        # (u_row, u_col, v_row, v_col, w_row, w_col, coeff)
        # M1
        (0, 0, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 0, 1), (0, 0, 1, 1, 0, 0, 1), (1, 1, 1, 1, 0, 0, 1),
        (0, 0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 1, 1, 1), (0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1),
        # M2
        (1, 0, 0, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0, 1),
        # M3
        (0, 0, 0, 1, 0, 1, 1), (0, 0, 1, 1, 0, 1, -1),
        # M4
        (1, 1, 1, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0, -1),
        # M5
        (0, 0, 1, 1, 0, 1, 1), (0, 1, 1, 1, 0, 1, 1),
        # M6
        (1, 0, 0, 0, 1, 1, 1), (0, 0, 0, 0, 1, 1, -1), (1, 0, 0, 1, 1, 1, 1), (0, 0, 0, 1, 1, 1, -1),
        # M7
        (0, 1, 1, 0, 0, 0, 1), (1, 1, 1, 0, 0, 0, -1), (0, 1, 1, 1, 0, 0, 1), (1, 1, 1, 1, 0, 0, -1)
    ]
    
    # This manual expansion is error-prone. I will use a standard loop for Strassen.
    
    # Standard Strassen for 2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    def add_strassen_block(a_r, a_c, b_r, b_c, w_r, w_c, offset):
        # a_r, a_c: top-left of A block (2x2)
        # b_r, b_c: top-left of B block (2x2)
        # w_r, w_c: top-left of W block (2x2)
        
        # M1
        idx = offset
        U[a_r][idx] += 1; U[a_r+1][idx] += 1
        V[b_r][idx] += 1; V[b_r+1][idx] += 1
        W[w_r][idx] += 1; W[w_r+1][idx] += 1
        
        # M2
        idx = offset + 1
        U[a_r+1][idx] += 1; U[a_r+1][idx] += 1 # A10, A11
        V[b_r][idx] += 1
        W[w_r+1][idx] += 1
        
        # M3
        idx = offset + 2
        U[a_r][idx] += 1
        V[b_r+1][idx] += 1; V[b_r+1][idx] -= 1 # B01, -B11
        W[w_r][idx] += 1
        
        # M4
        idx = offset + 3
        U[a_r+1][idx] += 1
        V[b_r+1][idx] += 1; V[b_r][idx] -= 1 # B10, -B00
        W[w_r+1][idx] += 1
        
        # M5
        idx = offset + 4
        U[a_r][idx] += 1; U[a_r][idx] += 1 # A00, A01
        V[b_r+1][idx] += 1
        W[w_r][idx] += 1; W[w_r+1][idx] += 1
        
        # M6
        idx = offset + 5
        U[a_r+1][idx] += 1; U[a_r][idx] -= 1 # A10, -A00
        V[b_r][idx] += 1; V[b_r+1][idx] += 1 # B00, B01
        W[w_r+1][idx] += 1
        
        # M7
        idx = offset + 6
        U[a_r][idx] += 1; U[a_r+1][idx] -= 1 # A01, -A11
        V[b_r+1][idx] += 1; V[b_r+1][idx] += 1 # B10, B11
        W[w_r][idx] += 1
        
    # Apply Strassen to 3 blocks
    # Block 0: A(0:2, 0:2), B(0:2, 0:2), W(0:2, 0:2)
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    # Block 1: A(0:2, 0:2), B(0:2, 2:4), W(0:2, 2:4)
    add_strassen_block(0, 0, 0, 2, 0, 2, 7)
    # Block 2: A(0:2, 0:2), B(0:2, 4:6), W(0:2, 4:6)
    add_strassen_block(0, 0, 0, 4, 0, 4, 14)
    
    # Part 2: Schoolbook for 2x3 * 3x3 blocks
    # A is 2x3 (rows 0-1, cols 2-4)
    # B blocks are 3x3 at cols 0:3 and 3:6? No, B is 5x6.
    # B[2:5, :] is 3x6.
    # Block 3: B[2:5, 0:3] (3x3)
    # Block 4: B[2:5, 3:6] (3x3)
    
    # Schoolbook for 2x3 * 3x3
    # A indices: i in 0..1, k in 0..2 (local) -> global row i, col 2+k
    # B indices: k in 0..2 (local), j in 0..2 (local) -> global row 2+k, col b_c + j
    # W indices: i in 0..1, j in 0..2 (local) -> global row w_c + j, col i?
    # W is p*n = 12x2. Index is k*n + i.
    # Here local i is row of A/C, local j is col of B/C.
    # Global W index: (w_c + j) * n + i.
    
    def add_schoolbook_2x3x3(a_r, a_c, b_r, b_c, w_c, offset):
        # a_r: 0, a_c: 2
        # b_r: 2, b_c: 0 or 3
        # w_c: 0 or 3
        for i in range(2):
            for k in range(3):
                for j in range(3):
                    idx = offset + i*9 + k*3 + j
                    U[a_r + i][idx] += 1
                    V[b_r + k][idx] += 1
                    W[w_c + j][idx] += 1 # Wait, W index is (col, row) or (row, col)?
                    # W[k*n+i, r].
                    # Here local j is col of C, local i is row of C.
                    # Global W row = (w_c + j) * n + i.
                    # So W[(w_c+j)*n + i][idx] += 1.
                    # My previous line was W[w_c+j][idx]. This is wrong if W is flattened.
                    # W is shape (p*n, rank).
                    # Index in W is row = (p_idx * n + i).
                    # So W[(w_c + j) * n + i][idx] += 1.
                    
    # Fix add_schoolbook_2x3x3
    def add_schoolbook_2x3x3(a_r, a_c, b_r, b_c, w_c, offset):
        for i in range(2):
            for k in range(3):
                for j in range(3):
                    idx = offset + i*9 + k*3 + j
                    U[a_r + i][idx] += 1
                    V[b_r + k][idx] += 1
                    W[(w_c + j) * n + i][idx] += 1

    # Block 3: A(0:2, 2:5), B(2:5, 0:3), W(0:3)
    add_schoolbook_2x3x3(0, 2, 2, 0, 0, 21)
    # Block 4: A(0:2, 2:5), B(2:5, 3:6), W(3:6)
    add_schoolbook_2x3x3(0, 2, 2, 3, 3, 39)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
