"""Matrix multiplication 2x3x8 via Strassen block decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return 7 (u,v,w) triples for 2x2x2 matrix multiply.
    u: (2,) input row indices, v: (2,) input col indices, w: (2,) output (row,col) -> (row*n+col)
    """
    # Standard Strassen 7 multiplications
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    return [
        # (u, v, w) where w is list of (out_row, out_col) with +/− coefficients
        (([1, 1], [0, 1]), ([1, 1], [0, 1]), [((0,0),1), ((1,1),1)]),
        (([1, 1], [1, 1]), ([1, 0], [0, 0]), [((1,0),1), ((1,1),-1)]),
        (([1], [0]), ([1], [1]), [((0,1),1), ((1,1),-1)]),
        (([1], [1]), ([1], [0]), [((0,0),-1), ((1,0),1)]),
        (([1, 1], [0, 1]), ([1], [1]), [((0,0),-1), ((0,1),1)]),
        (([1, -1], [1, 0]), ([1, 1], [0, 1]), [((1,1),1)]),
        (([1, -1], [1, 1]), ([1, 1], [1, 0]), [((0,0),1), ((1,1),-1)]),
    ]

def _schoolbook_2x1x2():
    """Return 4 (u,v,w) triples for 2x1x2 matrix multiply (2x1 times 1x2 -> 2x2).
    u: row index of 2x1, v: col index of 1x2, w: output (row,col)
    """
    return [
        (([1], [0]), ([1], [0]), [((0,0),1)]),
        (([1], [0]), ([1], [1]), [((0,1),1)]),
        (([1], [1]), ([1], [0]), [((1,0),1)]),
        (([1], [1]), ([1], [1]), [((1,1),1)]),
    ]

def construct():
    import time
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Split m=3 into m1=2 (Strassen) and m2=1 (schoolbook)
    m1, m2 = 2, 1
    # Split p=8 into p1=4 (two 2x2 blocks) and p2=0
    p1 = 4
    
    strassen = _strassen_2x2x2()
    school = _schoolbook_2x1x2()
    
    # Each of the 2 output column-pairs (2x2 blocks) gets:
    #   7 Strassen terms (for the 2x2x2 part) + 4 school terms (for 2x1x2 part) = 11
    # Total: 2 * 11 = 22... wait, p1=4 means 2 output pairs, each 2x2.
    # Actually p=8, so output is 2x8. We split into 4 pairs of 2 columns.
    # Each pair: 7 (Strassen on 2x2x2) + 4 (school on 2x1x2) = 11
    # Total: 4 * 11 = 44
    
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 6 x 44
    V = [[0]*rank for _ in range(m*p)]  # 24 x 44
    W = [[0]*rank for _ in range(p*n)]  # 16 x 44
    
    col = 0
    for out_block in range(4):  # 4 output column-pairs
        out_col_start = out_block * 2
        
        # Strassen part: 2x2x2
        for (u_spec, v_spec, w_spec) in strassen:
            # u_spec: (coeffs, indices) where indices are local row indices (0,1)
            # v_spec: (coeffs, indices) where indices are local col indices within the 2-col block
            u_coeffs, u_indices = u_spec
            v_coeffs, v_indices = v_spec
            for ui in range(len(u_indices)):
                row = u_indices[ui]
                U[row * m + out_col_start * 0 + (row % 2) * 0 + (row // 2) * 0][col] = 0  # placeholder
            # Actually, let me restructure. The u indices refer to rows of the 2x2 input.
            # In the full 2x3 matrix, row i (0 or 1) and col j (0,1,2).
            # The Strassen block uses cols 0,1. So U row index = i*m + j where j is in {0,1}.
            # v indices refer to cols of the 2x2 output, which maps to the output column pair.
            # In the full 3x8 matrix, the v part is row j (0,1,2) and col k.
            # The Strassen block uses input cols 0,1, so v row index = j*p + k where j in {0,1}, k in {out_col_start, out_col_start+1}.
            
            # Let me redo this more carefully.
            pass
        col += 7
        
        # School part: 2x1x2
        for (u_spec, v_spec, w_spec) in school:
            u_coeffs, u_indices = u_spec
            v_coeffs, v_indices = v_spec
            pass
        col += 4
    
    # The above approach is getting complex. Let me use a cleaner formulation.
    
    # Reset
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for out_block in range(4):
        out_col_start = out_block * 2
        
        # --- Strassen: 2x2x2 ---
        # u: row of 2x2 input -> U index = row * m + j where j in {0,1} (cols 0,1 of 2x3)
        # v: col of 2x2 input -> V index = j * p + k where j in {0,1}, k in {out_col_start, out_col_start+1}
        # w: output (r,c) -> W index = k * n + r where k = out_col_start + c, r in {0,1}
        
        for (u_c, u_i), (v_c, v_i), w_list in strassen:
            for a in range(len(u_i)):
                for b in range(len(v_i)):
                    u_row = u_i[a]
                    v_row = v_i[b]
                    U[u_row * m + 0 + (0)][col] += u_c[a]  # col j=0? No...
            # This indexing is wrong. Let me think again.
            pass
        col += 7
        
        # --- School: 2x1x2 ---
        for (u_c, u_i), (v_c, v_i), w_list in school:
            pass
        col += 4
    
    # I'm overcomplicating this. Let me write it out explicitly.
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for out_block in range(4):
        k0 = out_block * 2  # first output column in this block
        
        # Strassen 2x2x2: input rows 0,1; input cols 0,1 (of the 2x3); output cols k0, k0+1
        # u_local (row 0/1) -> U index: row * m + col_in_block
        # v_local (col 0/1) -> V index: col_in_block * p + (k0 + col_local)
        # w (out_row, out_col) -> W index: (k0 + out_col) * n + out_row
        
        S = _strassen_2x2x2()
        for entry in S:
            u_c, u_i = entry[0]
            v_c, v_i = entry[1]
            w_list = entry[2]
            # Add contributions to U, V, W
            for a in range(len(u_i)):
                U[u_i[a] * m + 0][col] += u_c[a]
                U[u_i[a] * m + 1][col] += 0  # cols 0 and 1 of 2x3
            # Wait, u_i are the local row indices (0 or 1), and the column in the 2x3 is determined by the block.
            # Actually for Strassen on 2x2x2, the input is 2x2, so:
            # U row = i (row of 2x3) * m + j (col of 2x3, which is 0 or 1 for Strassen block)
            # The u_i gives the row, but we also need the column. In the standard Strassen,
            # the input is 2x2, so u is (row, col) pair? No, u is just the row factor.
            # 
            # Let me reconsider. In tensor decomposition:
            # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
            # For 2x2x2: T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
            # U has 4 entries (i,j), V has 4 entries (j,k), W has 4 entries (i,k)
            # 
            # So u_i is a row index into U, which is i*m+j. But Strassen operates on the 2x2 input,
            # so the "rows" of U are (i,j) pairs where i in {0,1}, j in {0,1}.
            # 
            # I think my Strassen definition was wrong. Let me redefine.
            pass
        col += 7
        col += 4
    
    # This is getting too tangled. Let me just write it out completely explicitly.
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for out_block in range(4):
        k0 = out_block * 2
        
        # Strassen: input 2x2 (rows 0,1 of 2x3; cols 0,1 of 2x3)
        # U indices: (i, j) -> i*m + j, i in {0,1}, j in {0,1}
        # V indices: (j, k) -> j*p + k, j in {0,1}, k in {k0, k0+1}
        # W indices: (i, k) -> k*n + i, i in {0,1}, k in {k0, k0+1}
        
        # M1 = (A00+A11)(B00+B11)
        U[0*m+0][col] += 1; U[1*m+1][col] += 1
        V[0*p+k0][col] += 1; V[1*p+(k0+1)][col] += 1
        W[k0*n+0][col] += 1; W[(k0+1)*n+1][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*m+0][col] += 1; U[1*m+1][col] += 1
        V[0*p+k0][col] += 1
        W[(k0+1)*n+0][col] += 1; W[(k0+1)*n+1][col] += -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0*m+0][col] += 1
        V[0*p+(k0+1)][col] += 1; V[1*p+(k0+1)][col] += -1
        W[k0*n+1][col] += 1; W[(k0+1)*n+1][col] += -1
        col += 1
        
        # M4 = A11(B10-B00)
        U[1*m+1][col] += 1
        V[1*p+k0][col] += 1; V[0*p+k0][col] += -1
        W[k0*n+0][col] += -1; W[(k0+1)*n+0][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*m+0][col] += 1; U[0*m+1][col] += 1
        V[1*p+(k0+1)][col] += 1
        W[k0*n+0][col] += -1; W[k0*n+1][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1*m+0][col] += 1; U[0*m+0][col] += -1
        V[0*p+k0][col] += 1; V[0*p+(k0+1)][col] += 1
        W[(k0+1)*n+1][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0*m+1][col] += 1; U[1*m+1][col] += -1
        V[1*p+k0][col] += 1; V[1*p+(k0+1)][col] += 1
        W[k0*n+0][col] += 1; W[(k0+1)*n+1][col] += -1
        col += 1
        
        # School: 2x1x2, input row (0,1), input col 2 (of 2x3), output cols k0, k0+1
        # U indices: i*m + 2, i in {0,1}
        # V indices: 2*p + k, k in {k0, k0+1}
        # W indices: k*n + i, i in {0,1}, k in {k0, k0+1}
        
        # A0 * B0 -> C00
        U[0*m+2][col] += 1
        V[2*p+k0][col] += 1
        W[k0*n+0][col] += 1
        col += 1
        
        # A0 * B1 -> C01
        U[0*m+2][col] += 1
        V[2*p+(k0+1)][col] += 1
        W[(k0+1)*n+0][col] += 1
        col += 1
        
        # A1 * B0 -> C10
        U[1*m+2][col] += 1
        V[2*p+k0][col] += 1
        W[k0*n+1][col] += 1
        col += 1
        
        # A1 * B1 -> C11
        U[1*m+2][col] += 1
        V[2*p+(k0+1)][col] += 1
        W[(k0+1)*n+1][col] += 1
        col += 1
    
    # col should be 44
    assert col == rank, f"Expected {rank}, got {col}"
    
    # Convert to integers if possible
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
