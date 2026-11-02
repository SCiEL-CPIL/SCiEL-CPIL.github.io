"""Matrix multiplication 2x3 by 3x9 - rank-reduced decomposition using Strassen on 2x2 subblocks."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    # We'll build U, V, W matrices.
    # Strategy: split inner dim m=3 into (2,1). Split columns p=9 into groups of 2 and 1.
    # For a group of g columns, the subproblem is 2x3 by 3xg.
    # Split inner: A = [A2 | A1] where A2 is 2x2, A1 is 2x1
    # B = [B2; B1] where B2 is 2xg, B1 is 1xg
    # C = A2*B2 + A1*B1
    # A2*B2: 2x2 by 2xg. If g=2: Strassen rank 7. If g=1: schoolbook rank 4.
    # A1*B1: 2x1 by 1xg. Schoolbook rank 2*g.
    
    # Column grouping: 9 columns -> 4 groups of 2 + 1 group of 1
    # Groups: (0,1), (2,3), (4,5), (6,7), (8)
    groups = [(0,1), (2,3), (4,5), (6,7), (8,)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for cols in groups:
        g = len(cols)
        col_indices = list(cols)
        
        # Part 1: A2 (2x2) * B2 (2 x g)
        if g == 2:
            # Strassen for 2x2 * 2x2, rank 7
            # A2 = [[a00, a01], [a10, a11]]
            # B2 = [[b00, b01], [b10, b11]] (columns col_indices[0], col_indices[1])
            # Strassen factors:
            # M1 = (A00+A11)*(B00+B11)
            # M2 = (A10+A11)*B00
            # M3 = A00*(B10-B11)
            # M4 = A11*(B01-B10)
            # M5 = (A00+A01)*B11
            # M6 = (A10-A00)*(B00+B10)
            # M7 = (A01-A11)*(B01+B11)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # U factors (rows = 2x2=4, indexed as i*m+j in the 2x2 block)
            # A2 row indices in full A: row i, inner cols 0,1 -> i*3+0, i*3+1
            # B2 col indices in full B: inner rows 0,1, col k -> 0*3+k=0*3+k? No, B is m x p = 3x9
            # B row j (inner), col k: j*p + k = j*9 + k
            # W: output C[i,k], index k*n+i in W (trace convention W[k*n+i, r])
            
            # U for Strassen (4 rows x 7 cols)
            U_s = [
                # M1: A00+A11 -> row 0 (i=0,j=0) and row 3 (i=1,j=1)
                [1, 0, 0, 0, 0, 0, 0],  # M1
                [0, 0, 0, 0, 1, 0, 0],  # M2
                [0, 0, 1, 0, 0, 0, 0],  # M3
                [0, 0, 0, 1, 0, 0, 0],  # M4
                [0, 0, 0, 0, 0, 1, 0],  # M5 -- wait, M5 = (A00+A01)*B11
                # Let me redo this more carefully.
            ]
            
            # Let me define Strassen for 2x2*2x2 properly.
            # A = [[a0, a1], [a2, a3]] (row-major: a0=A00, a1=A01, a2=A10, a3=A11)
            # B = [[b0, b1], [b2, b3]] (row-major: b0=B00, b1=B01, b2=B10, b3=B11)
            # C = A*B
            # C00 = (a0+a3)(b0+b3) - (a2+a3)(b0+b2) + (a0+a1)(b3-b2) ... 
            # Standard Strassen:
            # P1 = (A00+A11)(B00+B11)
            # P2 = (A10+A11)(B00)
            # P3 = (A00)(B10-B11)
            # P4 = (A11)(B01-B10)
            # P5 = (A00+A01)(B11)
            # P6 = (A10-A00)(B00+B10)
            # P7 = (A01-A11)(B01+B11)
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # U (4x7): rows are A elements [A00, A01, A10, A11]
            U_s = [
                [1, 0, 1, 0, 1, 1, 0],  # A00: P1, P3, P5, P6
                [0, 0, 0, 0, 1, 0, 1],  # A01: P5, P7
                [0, 1, 0, 0, 0, 1, 0],  # A10: P2, P6
                [1, 1, 0, 1, 0, 0, 1],  # A11: P1, P2, P4, P7
            ]
            # V (4x7): rows are B elements [B00, B01, B10, B11]
            V_s = [
                [1, 1, 0, 0, 0, 1, 0],  # B00: P1, P2, P6
                [0, 0, 0, 1, 0, 0, 1],  # B01: P4, P7
                [0, 0, 1, 1, 0, 1, 0],  # B10: P3, P4, P6
                [1, 0, 0, 0, 1, 0, 1],  # B11: P1, P5, P7
            ]
            # W (4x7): rows are C elements [C00, C01, C10, C11]
            W_s = [
                [1, 0, 0, 1, 1, 0, 1],  # C00 = P1 + P4 - P5 + P7
                [0, 0, 1, 0, 1, 0, 0],  # C01 = P3 + P5
                [0, 1, 0, 1, 0, 0, 0],  # C10 = P2 + P4
                [1, 1, 1, 0, 0, 1, 0],  # C11 = P1 - P2 + P3 + P6 -- wait
            ]
            # Let me recheck: C11 = P1 - P2 + P3 + P6
            # So W for C11: [1, -1, 1, 0, 0, 1, 0]
            # But we need to verify. Actually standard Strassen:
            # C11 = P1 - P2 + P3 + P6
            # Yes.
            
            # Fix W:
            W_s = [
                [1, 0, 0, 1, -1, 0, 1],   # C00
                [0, 0, 1, 0, 1, 0, 0],    # C01
                [0, 1, 0, 1, 0, 0, 0],    # C10
                [1, -1, 1, 0, 0, 1, 0],   # C11
            ]
            
            # Now map to full indices.
            # U rows: A2 element (i, j_inner) where i in {0,1}, j_inner in {0,1}
            # Full A row index: i*3 + j_inner
            # V rows: B2 element (j_inner, col) where j_inner in {0,1}, col in col_indices
            # Full B row index: j_inner*9 + col
            # W rows: C element (i, col) where i in {0,1}, col in col_indices
            # Full W index: col*2 + i (trace convention W[k*n+i, r])
            
            rank_s = 7
            U_part = []
            V_part = []
            W_part = []
            
            # U: 4 rows (A2 elements) x 7 cols
            for ri in range(4):
                row_idx = (ri // 2) * 3 + (ri % 2)  # i = ri//2, j_inner = ri%2
                U_part.append([0] * rank_s)
                for c in range(rank_s):
                    U_part[-1][c] = U_s[ri][c]
            
            # V: 4 rows (B2 elements) x 7 cols
            for ri in range(4):
                j_inner = ri // 2  # 0 or 1
                col_local = ri % 2  # 0 or 1
                col_global = col_indices[col_local]
                row_idx = j_inner * 9 + col_global
                V_part.append([0] * rank_s)
                for c in range(rank_s):
                    V_part[-1][c] = V_s[ri][c]
            
            # W: 4 rows (C elements) x 7 cols
            for ri in range(4):
                i = ri // 2  # 0 or 1
                col_local = ri % 2  # 0 or 1
                col_global = col_indices[col_local]
                row_idx = col_global * 2 + i
                W_part.append([0] * rank_s)
                for c in range(rank_s):
                    W_part[-1][c] = W_s[ri][c]
            
            U_cols.append(U_part)
            V_cols.append(V_part)
            W_cols.append(W_part)
        else:
            # g == 1: schoolbook for 2x2 * 2x1, rank 4
            # A2 = [[a0, a1], [a2, a3]], B2 = [[b0], [b1]] (single column)
            # C00 = a0*b0 + a1*b1
            # C10 = a2*b0 + a3*b1
            col_global = col_indices[0]
            rank_s = 4
            U_part = []
            V_part = []
            W_part = []
            
            # U: A2 elements [A00, A01, A10, A11]
            for ri in range(4):
                i = ri // 2
                j_inner = ri % 2
                row_idx = i * 3 + j_inner
                U_part.append([0] * rank_s)
            
            # V: B2 elements [B00, B10] (only 2, but we have 4 rows in U)
            # Actually for schoolbook 2x2 * 2x1:
            # C00 = A00*B00 + A01*B10
            # C10 = A10*B00 + A11*B10
            # So we need 4 terms.
            # U rows: [A00, A01, A10, A11], V rows: [B00, B10, B00, B10] (repeated)
            # W rows: [C00, C10, C00, C10]
            
            U_s = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            V_s = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]  # Wait, this is wrong
            # Let me redo:
            # Term 0: A00 * B00 -> U[0][0]=1, V[0][0]=1
            # Term 1: A01 * B10 -> U[1][1]=1, V[1][1]=1
            # Term 2: A10 * B00 -> U[2][2]=1, V[2][2]=1
            # Term 3: A11 * B10 -> U[3][3]=1, V[3][3]=1
            # But V has only 2 rows (B00, B10), so:
            # V row 0 (B00): [1, 0, 1, 0]
            # V row 1 (B10): [0, 1, 0, 1]
            # W: C00 = term0 + term1, C10 = term2 + term3
            # W row 0 (C00): [1, 1, 0, 0]
            # W row 1 (C10): [0, 0, 1, 1]
            
            # U: 4 rows x 4 cols
            U_s = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            # V: 2 rows x 4 cols
            V_s = [[1,0,1,0],[0,1,0,1]]
            # W: 2 rows x 4 cols
            W_s = [[1,1,0,0],[0,0,1,1]]
            
            # Map U rows to full A indices
            for ri in range(4):
                i = ri // 2
                j_inner = ri % 2
                row_idx = i * 3 + j_inner
                U_part.append(U_s[ri])
            
            # Map V rows to full B indices
            for ri in range(2):
                j_inner = ri  # 0 or 1
                row_idx = j_inner * 9 + col_global
                V_part.append(V_s[ri])
            
            # Map W rows to full W indices
            for ri in range(2):
                i = ri  # 0 or 1
                row_idx = col_global * 2 + i
                W_part.append(W_s[ri])
            
            U_cols.append(U_part)
            V_cols.append(V_part)
            W_cols.append(W_part)
        
        # Part 2: A1 (2x1) * B1 (1 x g)
        # A1 = [[a0], [a1]] (inner row 2, cols 0,1 in A -> actually A[:, 2] but A is 2x3)
        # Wait, A is 2x3. A1 is the 2x1 submatrix from inner col 2: A[:, 2]
        # B1 is the 1xg submatrix from inner row 2: B[2, col_indices]
        # C1 = A1 * B1: 2x1 * 1xg = 2xg
        # C1[i, col] = A[i, 2] * B[2, col]
        # Schoolbook: rank 2*g
        rank_t = 2 * g
        U_part = []
        V_part = []
        W_part = []
        
        for col_local, col_global in enumerate(col_indices):
            # Term for C[0, col_global]: A[0,2] * B[2, col_global]
            # U row for A[0,2]: row index 0*3+2 = 2
            U_part.append([0] * rank_t)
            U_part[-1][col_local] = 1
            # U row for A[1,2]: row index 1*3+2 = 5
            U_part.append([0] * rank_t)
            U_part[-1][col_local + g] = 1
            # V row for B[2, col_global]: row index 2*9 + col_global
            V_part.append([0] * rank_t)
            V_part[-1][col_local] = 1
            V_part[-1][col_local + g] = 1
            # W row for C[0, col_global]: index col_global*2 + 0
            W_part.append([0] * rank_t)
            W_part[-1][col_local] = 1
            # W row for C[1, col_global]: index col_global*2 + 1
            W_part.append([0] * rank_t)
            W_part[-1][col_local + g] = 1
        
        U_cols.append(U_part)
        V_cols.append(V_part)
        W_cols.append(W_part)
    
    # Now combine all parts. Each part has different row counts and rank.
    # We need to concatenate along rank (columns) and pad rows.
    
    # Total rows: U has n*m = 6 rows, V has m*p = 27 rows, W has p*n = 18 rows
    # Collect all U rows, V rows, W rows by their full index.
    
    U_full = {i: [] for i in range(6)}
    V_full = {i: [] for i in range(27)}
    W_full = {i: [] for i in range(18)}
    
    for U_part, V_part, W_part in zip(U_cols, V_cols, W_cols):
        rank_part = len(U_part[0])
        # U_part rows are in order of A2 elements then A1 elements
        # We need to track which full row index each part row corresponds to.
        # This is getting complex. Let me restructure.
        pass
    
    # Let me restructure the whole thing to be cleaner.
    
    U_rows = {}
    V_rows = {}
    W_rows = {}
    rank_offset = 0
    
    for cols in groups:
        g = len(cols)
        col_indices = list(cols)
        
        # Part 1: A2 * B2
        if g == 2:
            rank_s = 7
            U_s = [
                [1, 0, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, 1],
            ]
            V_s = [
                [1, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 1],
            ]
            W_s = [
                [1, 0, 0, 1, -1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [1, -1, 1, 0, 0, 1, 0],
            ]
            # U rows: A2[i, j] for i in 0,1; j in 0,1
            for ri in range(4):
                i = ri // 2
                j = ri % 2
                row_idx = i * 3 + j
                if row_idx not in U_rows:
                    U_rows[row_idx] = []
                U_rows[row_idx].extend(U_s[ri])
            # V rows: B2[j, col] for j in 0,1; col in col_indices
            for ri in range(4):
                j = ri // 2
                col_local = ri % 2
                col_global = col_indices[col_local]
                row_idx = j * 9 + col_global
                if row_idx not in V_rows:
                    V_rows[row_idx] = []
                V_rows[row_idx].extend(V_s[ri])
            # W rows: C[i, col] for i in 0,1; col in col_indices
            for ri in range(4):
                i = ri // 2
                col_local = ri % 2
                col_global = col_indices[col_local]
                row_idx = col_global * 2 + i
                if row_idx not in W_rows:
                    W_rows[row_idx] = []
                W_rows[row_idx].extend(W_s[ri])
            rank_offset += rank_s
        else:
            rank_s = 4
            col_global = col_indices[0]
            U_s = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            V_s = [[1,0,1,0],[0,1,0,1]]
            W_s = [[1,1,0,0],[0,0,1,1]]
            for ri in range(4):
                i = ri // 2
                j = ri % 2
                row_idx = i * 3 + j
                if row_idx not in U_rows:
                    U_rows[row_idx] = []
                U_rows[row_idx].extend(U_s[ri])
            for ri in range(2):
                j = ri
                row_idx = j * 9 + col_global
                if row_idx not in V_rows:
                    V_rows[row_idx] = []
                V_rows[row_idx].extend(V_s[ri])
            for ri in range(2):
                i = ri
                row_idx = col_global * 2 + i
                if row_idx not in W_rows:
                    W_rows[row_idx] = []
                W_rows[row_idx].extend(W_s[ri])
            rank_offset += rank_s
        
        # Part 2: A1 * B1
        rank_t = 2 * g
        for col_local, col_global in enumerate(col_indices):
            # A[0,2] * B[2, col_global] -> C[0, col_global]
            row_u1 = 0 * 3 + 2  # = 2
            row_v1 = 2 * 9 + col_global
            row_w1 = col_global * 2 + 0
            if row_u1 not in U_rows:
                U_rows[row_u1] = [0] * (rank_offset + rank_t)
            if row_v1 not in V_rows:
                V_rows[row_v1] = [0] * (rank_offset + rank_t)
            if row_w1 not in W_rows:
                W_rows[row_w1] = [0] * (rank_offset + rank_t)
            U_rows[row_u1][rank_offset + col_local] = 1
            V_rows[row_v1][rank_offset + col_local] = 1
            W_rows[row_w1][rank_offset + col_local] = 1
            
            # A[1,2] * B[2, col_global] -> C[1, col_global]
            row_u2 = 1 * 3 + 2  # = 5
            row_w2 = col_global * 2 + 1
            if row_u2 not in U_rows:
                U_rows[row_u2] = [0] * (rank_offset + rank_t)
            if row_w2 not in W_rows:
                W_rows[row_w2] = [0] * (rank_offset + rank_t)
            U_rows[row_u2][rank_offset + col_local + g] = 1
            V_rows[row_v1][rank_offset + col_local + g] = 1
            W_rows[row_w2][rank_offset + col_local + g] = 1
        
        rank_offset += rank_t
    
    total_rank = rank_offset
    U = []
    for i in range(6):
        if i in U_rows:
            U.append(U_rows[i])
        else:
            U.append([0] * total_rank)
    V = []
    for i in range(27):
        if i in V_rows:
            V.append(V_rows[i])
        else:
            V.append([0] * total_rank)
    W = []
    for i in range(18):
        if i in W_rows:
            W.append(W_rows[i])
        else:
            W.append([0] * total_rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
