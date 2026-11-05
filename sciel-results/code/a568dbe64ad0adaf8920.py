"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes:
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (15, rank)
    # W: (p*n, rank) = (10, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        U.append([0] * 1) # Placeholder, will resize later or build lists of lists
        # Actually, building column by column is easier with lists of columns, but 
        # the interface requires U[i][r]. 
        # So U is list of rows. 
        # Let's build columns first, then transpose.
        pass

    # It is easier to build columns as lists of (row, val) pairs, then construct the matrix.
    cols_U = []
    cols_V = []
    cols_W = []
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        cols_U.append({u_row: u_val})
        cols_V.append({v_row: v_val})
        cols_W.append({w_row: w_val})

    def add_strassen(i0, j0, k0):
        # A indices: i in {i0, i0+1}, j in {j0, j0+1}
        # B indices: j in {j0, j0+1}, k in {k0, k0+1}
        # C indices: i in {i0, i0+1}, k in {k0, k0+1}
        
        # M1 = (a00+a11)(b00+b11)
        add_term(i0*m+j0, j0*p+k0, k0*n+i0, 1, 1, 1) # c00
        add_term(i0*m+j0, j0*p+k0, (k0+1)*n+i0+1, 1, 1, 1) # c11
        add_term(i0*m+j0+1, j0*p+k0+1, k0*n+i0, 1, 1, 1) # c00
        add_term(i0*m+j0+1, j0*p+k0+1, (k0+1)*n+i0+1, 1, 1, 1) # c11
        
        # Wait, the add_term above adds a single entry to the column.
        # For Strassen, one multiplication term contributes to multiple W entries.
        # So I need to add multiple W entries for the same U/V column.
        
        # Let's redefine add_term to allow multiple W entries.
        pass

    # Redefine helper to handle multiple W entries
    def add_term_multi_w(u_row, v_row, w_entries, u_val, v_val):
        # w_entries: list of (w_row, w_val)
        cols_U.append({u_row: u_val})
        cols_V.append({v_row: v_val})
        cols_W.append({w_row: w_val for w_row, w_val in w_entries})

    def add_strassen(i0, j0, k0):
        # a00 = U[i0*m+j0], a01 = U[i0*m+j0+1]
        # a10 = U[(i0+1)*m+j0], a11 = U[(i0+1)*m+j0+1]
        # b00 = V[j0*p+k0], b01 = V[j0*p+k0+1]
        # b10 = V[(j0+1)*p+k0], b11 = V[(j0+1)*p+k0+1]
        
        # M1 = (a00+a11)(b00+b11)
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: c00+=1, c11+=1
        # We need to accumulate W values.
        # Since we are building columns, each column is one multiplication.
        # The W matrix entry for this column is the sum of contributions.
        
        # This approach of "add_term" appending to a list of dicts is fine,
        # but we must ensure that for a single column, the W dict has the correct sum.
        # In Strassen, one column corresponds to one M_k.
        # M1 contributes to c00 and c11.
        
        # So for M1:
        # U col: {a00: 1, a11: 1}
        # V col: {b00: 1, b11: 1}
        # W col: {c00: 1, c11: 1}
        
        # But my add_term_multi_w only takes one u_row and one v_row.
        # Strassen M1 has two u_rows and two v_rows.
        
        # So I need a general add_column(u_dict, v_dict, w_dict)
        
        pass

    def add_column(u_dict, v_dict, w_dict):
        cols_U.append(u_dict)
        cols_V.append(v_dict)
        cols_W.append(w_dict)

    def add_strassen(i0, j0, k0):
        # Indices
        u_00 = i0*m + j0
        u_01 = i0*m + j0 + 1
        u_10 = (i0+1)*m + j0
        u_11 = (i0+1)*m + j0 + 1
        
        v_00 = j0*p + k0
        v_01 = j0*p + k0 + 1
        v_10 = (j0+1)*p + k0
        v_11 = (j0+1)*p + k0 + 1
        
        w_00 = k0*n + i0
        w_01 = k0*n + i0 + 1
        w_10 = (k0+1)*n + i0
        w_11 = (k0+1)*n + i0 + 1
        
        # M1 = (a00+a11)(b00+b11)
        add_column({u_00: 1, u_11: 1}, {v_00: 1, v_11: 1}, {w_00: 1, w_11: 1})
        
        # M2 = (a10+a11)b00
        add_column({u_10: 1, u_11: 1}, {v_00: 1}, {w_10: 1})
        
        # M3 = a00(b01-b11)
        add_column({u_00: 1}, {v_01: 1, v_11: -1}, {w_01: 1})
        
        # M4 = a11(b10-b00)
        add_column({u_11: 1}, {v_10: 1, v_00: -1}, {w_00: 1, w_10: 1})
        
        # M5 = (a00+a01)b11
        add_column({u_00: 1, u_01: 1}, {v_11: 1}, {w_00: -1, w_01: 1})
        
        # M6 = (a10-a00)(b00+b01)
        add_column({u_10: 1, u_00: -1}, {v_00: 1, v_01: 1}, {w_11: 1})
        
        # M7 = (a01-a11)(b10+b11)
        add_column({u_01: 1, u_11: -1}, {v_10: 1, v_11: 1}, {w_00: 1})

    def add_schoolbook_2x2x1(i0, j0, k0):
        # A (2x2) x B (2x1) -> C (2x1)
        # c00 = a00*b00 + a01*b10
        # c10 = a10*b00 + a11*b10
        
        u_00 = i0*m + j0
        u_01 = i0*m + j0 + 1
        u_10 = (i0+1)*m + j0
        u_11 = (i0+1)*m + j0 + 1
        
        v_00 = j0*p + k0
        v_10 = (j0+1)*p + k0
        
        w_00 = k0*n + i0
        w_10 = (k0+1)*n + i0 # Wait, k is fixed. 
        # C indices: i in {i0, i0+1}, k = k0.
        # w_00 = k0*n + i0
        # w_10 = k0*n + i0 + 1
        
        # Term 1: a00*b00 -> c00
        add_column({u_00: 1}, {v_00: 1}, {w_00: 1})
        
        # Term 2: a01*b10 -> c00
        add_column({u_01: 1}, {v_10: 1}, {w_00: 1})
        
        # Term 3: a10*b00 -> c10
        add_column({u_10: 1}, {v_00: 1}, {w_10: 1})
        
        # Term 4: a11*b10 -> c10
        add_column({u_11: 1}, {v_10: 1}, {w_10: 1})

    def add_schoolbook_2x1x5(i0, j0, k0):
        # A (2x1) x B (1x5) -> C (2x5)
        # c0k = a00*b0k
        # c1k = a10*b0k
        
        u_00 = i0*m + j0
        u_10 = (i0+1)*m + j0
        
        for k in range(5):
            v_0k = j0*p + (k0 + k)
            w_0k = (k0 + k)*n + i0
            w_1k = (k0 + k)*n + i0 + 1
            
            # Term: a00*b0k -> c0k
            add_column({u_00: 1}, {v_0k: 1}, {w_0k: 1})
            
            # Term: a10*b0k -> c1k
            add_column({u_10: 1}, {v_0k: 1}, {w_1k: 1})

    # Execute decomposition
    # 1. Strassen 2x2x2 for A[0:2, 0:2] x B[0:2, 0:2]
    add_strassen(0, 0, 0)
    
    # 2. Strassen 2x2x2 for A[0:2, 0:2] x B[0:2, 2:4]
    add_strassen(0, 0, 2)
    
    # 3. Schoolbook 2x2x1 for A[0:2, 0:2] x B[0:2, 4]
    add_schoolbook_2x2x1(0, 0, 4)
    
    # 4. Schoolbook 2x1x5 for A[0:2, 2] x B[2, 0:5]
    add_schoolbook_2x1x5(0, 2, 0)

    # Convert columns to matrices
    rank = len(cols_U)
    U_mat = [[0] * rank for _ in range(n*m)]
    V_mat = [[0] * rank for _ in range(m*p)]
    W_mat = [[0] * rank for _ in range(p*n)]
    
    for r in range(rank):
        for row, val in cols_U[r].items():
            U_mat[row][r] = val
        for row, val in cols_V[r].items():
            V_mat[row][r] = val
        for row, val in cols_W[r].items():
            W_mat[row][r] = val
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
