"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{2,2,15} into 7 blocks of M_{2,2,2} and 1 block of M_{2,2,1}.
    # 7 * 7 + 1 * 4 = 53.
    # The baseline is 2 * 2 * 15 = 60.
    
    rank_total = 0
    # Pre-allocate lists for U, V, W
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a block
    # Local indices: i in 0..n-1, j in 0..m-1, k in 0..p_block-1
    # Global indices: i_global = i, j_global = j, k_global = offset + k
    # U is (n*m, rank). Index row: i*m + j
    # V is (m*p, rank). Index row: j*p + k
    # W is (p*n, rank). Index row: k*n + i
    
    # Strassen decomposition for 2x2x2
    # Rank 7
    # T = (A B) (C D)
    # M1 = (A+B)(C+D)
    # M2 = (A+B)D
    # M3 = A(C-D)
    # M4 = BD
    # M5 = (A-D)(C+B)
    # M6 = (C-D)B
    # M7 = (A+B)(D-C) -- wait, standard Strassen is:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's map this to U, V, W.
    # U[r, col] * V[s, col] * W[t, col]
    # U corresponds to A, V to B, W to C (output).
    # A is 2x2. Rows indexed by (i,j). i in 0..1, j in 0..1.
    # A00: i=0,j=0. A01: i=0,j=1. A10: i=1,j=0. A11: i=1,j=1.
    # B is 2x2. Rows indexed by (j,k). j in 0..1, k in 0..1.
    # B00: j=0,k=0. B01: j=0,k=1. B10: j=1,k=0. B11: j=1,k=1.
    # C is 2x2. Rows indexed by (k,i). k in 0..1, i in 0..1.
    # C00: k=0,i=0. C01: k=0,i=1. C10: k=1,i=0. C11: k=1,i=1.
    
    # U values for each term (rank 7):
    # Term 0: P1 = (A00 + A11)(B00 + B11)
    # A part: A00(1) + A11(1). U[0,0]=1, U[3,0]=1 (row 3 is i=1,j=1)
    # B part: B00(1) + B11(1). V[0,0]=1, V[3,0]=1 (row 3 is j=1,k=1)
    # C part: C00(1) + C11(1) - ... wait.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's list the 7 products and their contributions to C.
    # 1. P1 = (A00+A11)(B00+B11) -> C00: +1, C11: +1
    # 2. P2 = (A10+A11)B00 -> C10: +1, C11: -1
    # 3. P3 = A00(B01-B11) -> C01: +1, C11: +1
    # 4. P4 = A11(B10-B00) -> C00: +1, C10: +1
    # 5. P5 = (A00+A01)B11 -> C00: -1, C01: +1
    # 6. P6 = (A10-A00)(B00+B01) -> C11: +1
    # 7. P7 = (A01-A11)(B10+B11) -> C00: +1
    
    # U (4x7)
    # Col 0 (P1): A00=1, A11=1
    # Col 1 (P2): A10=1, A11=1
    # Col 2 (P3): A00=1
    # Col 3 (P4): A11=1
    # Col 4 (P5): A00=1, A01=1
    # Col 5 (P6): A10=1, A00=-1
    # Col 6 (P7): A01=1, A11=-1
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # i=0,j=0 (A00)
        [0, 0, 0, 0, 1, 0, 1],  # i=0,j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0 (A10)
        [1, 1, 0, 1, 0, 0, -1]  # i=1,j=1 (A11)
    ]
    
    # V (4x7)
    # Col 0 (P1): B00=1, B11=1
    # Col 1 (P2): B00=1
    # Col 2 (P3): B01=1, B11=-1
    # Col 3 (P4): B10=1, B00=-1
    # Col 4 (P5): B11=1
    # Col 5 (P6): B00=1, B01=1
    # Col 6 (P7): B10=1, B11=1
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # j=0,k=0 (B00)
        [0, 0, 1, 0, 0, 1, 0],  # j=0,k=1 (B01)
        [0, 0, 0, 1, 0, 0, 1],  # j=1,k=0 (B10)
        [1, 0, -1, 0, 1, 0, 1]  # j=1,k=1 (B11)
    ]
    
    # W (4x7)
    # Col 0 (P1): C00=1, C11=1
    # Col 1 (P2): C10=1, C11=-1
    # Col 2 (P3): C01=1, C11=1
    # Col 3 (P4): C00=1, C10=1
    # Col 4 (P5): C00=-1, C01=1
    # Col 5 (P6): C11=1
    # Col 6 (P7): C00=1
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # k=0,i=0 (C00)
        [0, 0, 1, 0, 1, 0, 0],  # k=0,i=1 (C01)
        [0, 1, 0, 1, 0, 0, 0],  # k=1,i=0 (C10)
        [1, -1, 1, 0, 0, 1, 0]  # k=1,i=1 (C11)
    ]

    # Standard decomposition for 2x2x1 (rank 4)
    # A is 2x2, B is 2x1, C is 2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # U (4x4):
    # Col 0: A00=1
    # Col 1: A01=1
    # Col 2: A10=1
    # Col 3: A11=1
    U_221 = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    
    # V (2x4):
    # Col 0: B0=1
    # Col 1: B1=1
    # Col 2: B0=1
    # Col 3: B1=1
    V_221 = [
        [1, 0, 1, 0], # B0 (j=0)
        [0, 1, 0, 1]  # B1 (j=1)
    ]
    
    # W (2x4):
    # Col 0: C0=1
    # Col 1: C0=1
    # Col 2: C1=1
    # Col 3: C1=1
    W_221 = [
        [1, 1, 0, 0], # C0 (i=0)
        [0, 0, 1, 1]  # C1 (i=1)
    ]

    # Assemble global matrices
    # We will store the columns in a list, then combine.
    # Since we need to return lists of lists, we can just build the rows dynamically.
    # But it's easier to build the full U, V, W matrices as we go.
    
    total_rank = 7 * 7 + 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Add 7 Strassen blocks
    for b in range(7):
        # k_offset = 2 * b
        # For each local column r in 0..6:
        for r in range(7):
            col = col_offset + r
            
            # U contribution
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] += U_strassen[i * m + j][r]
            
            # V contribution
            for j in range(m):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    V[j * p + k_global][col] += V_strassen[j * 2 + k_local][r]
            
            # W contribution
            for k_local in range(2):
                k_global = 2 * b + k_local
                for i in range(n):
                    W[k_global * n + i][col] += W_strassen[k_local * n + i][r]
        
        col_offset += 7

    # Add 1 standard block for k = 14
    k_global = 14
    for r in range(4):
        col = col_offset + r
        
        # U contribution
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] += U_221[i * m + j][r]
        
        # V contribution
        for j in range(m):
            V[j * p + k_global][col] += V_221[j][r]
        
        # W contribution
        for i in range(n):
            W[k_global * n + i][col] += W_221[i][r]
    
    col_offset += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
