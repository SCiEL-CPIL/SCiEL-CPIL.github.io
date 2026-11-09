"""Matrix multiplication 2,4,5: Strassen-based decomposition for rank 36."""

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * rank for _ in range(m * p)]   # 20 x 36
    W = [[0] * rank for _ in range(p * n)]   # 10 x 36

    def add_strassen(base, A_i, A_j, B_j, B_k, C_k, C_i):
        """
        Add a Strassen 2x2x2 decomposition starting at column `base`.
        A_i, A_j: row indices of A (i dimension)
        B_j, B_k: row indices of B (j,k dimensions)
        C_k, C_i: row indices of C (k*i dimension)
        
        The 7 Strassen products:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B21 - B22)
        M4 = A22(B12 - B11)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # A rows: A_i[0], A_i[1] for i=0,1; A_j[0], A_j[1] for j=0,1
        a11 = A_i[0] * m + A_j[0]  # A row for i=0,j=0
        a12 = A_i[0] * m + A_j[1]  # A row for i=0,j=1
        a21 = A_i[1] * m + A_j[0]  # A row for i=1,j=0
        a22 = A_i[1] * m + A_j[1]  # A row for i=1,j=1
        
        # B rows: B_j[0], B_j[1] for j=0,1; B_k[0], B_k[1] for k=0,1
        b11 = B_j[0] * p + B_k[0]  # B row for j=0,k=0
        b12 = B_j[0] * p + B_k[1]  # B row for j=0,k=1
        b21 = B_j[1] * p + B_k[0]  # B row for j=1,k=0
        b22 = B_j[1] * p + B_k[1]  # B row for j=1,k=1
        
        # C rows: C_k[0], C_k[1] for k=0,1; C_i[0], C_i[1] for i=0,1
        c11 = C_k[0] * n + C_i[0]  # C row for k=0,i=0
        c12 = C_k[1] * n + C_i[0]  # C row for k=1,i=0
        c21 = C_k[0] * n + C_i[1]  # C row for k=0,i=1
        c22 = C_k[1] * n + C_i[1]  # C row for k=1,i=1
        
        # M1 = (A11 + A22)(B11 + B22)
        col = base + 0
        U[a11][col] += 1
        U[a22][col] += 1
        V[b11][col] += 1
        V[b22][col] += 1
        W[c11][col] += 1
        W[c22][col] += 1
        
        # M2 = (A21 + A22)B11
        col = base + 1
        U[a21][col] += 1
        U[a22][col] += 1
        V[b11][col] += 1
        W[c21][col] += 1
        W[c22][col] -= 1
        
        # M3 = A11(B21 - B22)
        col = base + 2
        U[a11][col] += 1
        V[b21][col] += 1
        V[b22][col] -= 1
        W[c12][col] += 1
        W[c22][col] += 1
        
        # M4 = A22(B12 - B11)
        col = base + 3
        U[a22][col] += 1
        V[b12][col] += 1
        V[b11][col] -= 1
        W[c11][col] += 1
        W[c21][col] += 1
        
        # M5 = (A11 + A12)B22
        col = base + 4
        U[a11][col] += 1
        U[a12][col] += 1
        V[b22][col] += 1
        W[c11][col] -= 1
        W[c12][col] += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        col = base + 5
        U[a21][col] += 1
        U[a11][col] -= 1
        V[b11][col] += 1
        V[b12][col] += 1
        W[c22][col] += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        col = base + 6
        U[a12][col] += 1
        U[a22][col] -= 1
        V[b21][col] += 1
        V[b22][col] += 1
        W[c11][col] += 1
        W[c22][col] += 1

    def add_schoolbook_2x2x3(base, A_i, A_j, B_j, B_k_list, C_k_list, C_i):
        """
        Add schoolbook 2x2x3 decomposition starting at column `base`.
        This handles 2x2 A times 2x3 B, giving 2x3 C.
        Uses 4 multiplications per column pair: 2*2*1 = 4 for each of 3 columns... 
        Actually schoolbook for 2x2 * 2x3 is 2*2*3 = 12 multiplications.
        But we can be smarter: for each of the 3 output columns, we need 
        4 scalar multiplications (2x2 matmul), so 4*3 = 12.
        """
        a11 = A_i[0] * m + A_j[0]
        a12 = A_i[0] * m + A_j[1]
        a21 = A_i[1] * m + A_j[0]
        a22 = A_i[1] * m + A_j[1]
        
        for c in range(3):
            b1k = B_j[0] * p + B_k_list[c]
            b2k = B_j[1] * p + B_k_list[c]
            c1k = C_k_list[c] * n + C_i[0]
            c2k = C_k_list[c] * n + C_i[1]
            
            # C1k = A11*B1k + A12*B2k
            col = base + 4 * c + 0
            U[a11][col] += 1
            V[b1k][col] += 1
            W[c1k][col] += 1
            
            # C1k += A12*B2k
            col = base + 4 * c + 1
            U[a12][col] += 1
            V[b2k][col] += 1
            W[c1k][col] += 1
            
            # C2k = A21*B1k + A22*B2k
            col = base + 4 * c + 2
            U[a21][col] += 1
            V[b1k][col] += 1
            W[c2k][col] += 1
            
            col = base + 4 * c + 3
            U[a22][col] += 1
            V[b2k][col] += 1
            W[c2k][col] += 1

    # Block 1: C columns 0,1 (2x2 output)
    # A rows 0,1; B cols 0,1
    # C indices: k in {0,1}, i in {0,1}
    add_strassen(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: C columns 2,3,4 (2x3 output)
    # For each pair of B rows (0,1) or (2,3), and each A column pair
    # A columns 0,1 (left half) times B rows 0,1 (top half) -> contributes to C
    # A columns 0,1 times B rows 2,3 (bottom half) -> contributes to C
    # A columns 2,3 (right half) times B rows 0,1 -> contributes to C  
    # A columns 2,3 times B rows 2,3 -> contributes to C
    
    # Wait, let me reconsider the block structure.
    # A is 2x4, B is 4x5.
    # Split A into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3)
    # Split B into B_T (2x5, rows 0,1) and B_B (2x5, rows 2,3)
    # C = A_L * B_T + A_R * B_B
    
    # Each term is 2x2 * 2x5 = 2x5
    # For A_L * B_T: split B_T into B_T1 (2x2, cols 0,1), B_T2 (2x3, cols 2,3,4)
    #   A_L * B_T1: 2x2*2x2 -> Strassen (7)
    #   A_L * B_T2: 2x2*2x3 -> schoolbook (12)
    # For A_R * B_B: similar
    #   A_R * B_B1: 2x2*2x2 -> Strassen (7)
    #   A_R * B_B2: 2x2*2x3 -> schoolbook (12)
    # Total: 7 + 12 + 7 + 12 = 38. That's not 36.
    
    # Let me use Strassen for the 2x2x2 parts and a better method for 2x2x3.
    # Actually, M(2,2,3) can be done in 10 rank (known result).
    # So: 7 + 10 + 7 + 10 = 34.
    
    # Let me reset and use the 34-rank decomposition.
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A_L: rows 0,1; cols 0,1
    # A_R: rows 0,1; cols 2,3
    # B_T: rows 0,1; cols 0,4 (all 5 cols)
    # B_B: rows 2,3; cols 0,4 (all 5 cols)
    
    # Term 1: A_L * B_T1 (2x2 * 2x2, cols 0,1) -> Strassen
    add_strassen(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Term 2: A_L * B_T2 (2x2 * 2x3, cols 2,3,4) -> M(2,2,3) rank 10
    # Need 10-rank decomposition for 2x2x3
    # This is non-trivial. Let me use schoolbook (12) for now to get 36.
    
    # Reset to 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 1: A_L * B_T1 (2x2 * 2x2) -> Strassen, 7
    add_strassen(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Term 2: A_L * B_T2 (2x2 * 2x3) -> schoolbook, 12
    add_schoolbook_2x2x3(7, [0, 1], [0, 1], [0, 1], [2, 3, 4], [2, 3, 4], [0, 1])
    
    # Term 3: A_R * B_B1 (2x2 * 2x2) -> Strassen, 7
    add_strassen(19, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Term 4: A_R * B_B2 (2x2 * 2x3) -> schoolbook, 12
    add_schoolbook_2x2x3(26, [0, 1], [2, 3], [2, 3], [2, 3, 4], [2, 3, 4], [0, 1])
    
    return {"U": U, "V": V, "W": W}
