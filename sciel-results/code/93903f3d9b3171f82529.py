"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 117
    rank = 117
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: U[i][col] += u, V[j][col] += v, W[k][col] += w
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Helper to apply Strassen for a 2x2 block
    # Local indices 0,1 map to global offsets
    # A block at (i0, i1) x (j0, j1)
    # B block at (j0, j1) x (k0, k1)
    # C block at (i0, i1) x (k0, k1)
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # Local indices:
        # a00 = U[i0*m + j0]
        # a01 = U[i0*m + j1]
        # a10 = U[i1*m + j0]
        # a11 = U[i1*m + j1]
        
        # b00 = V[j0*p + k0]
        # b01 = V[j0*p + k1]
        # b10 = V[j1*p + k0]
        # b11 = V[j1*p + k1]
        
        # c00 = W[k0*n + i0]
        # c01 = W[k1*n + i0]
        # c10 = W[k0*n + i1]
        # c11 = W[k1*n + i1]
        
        # 1. m1 = (a00+a01)(b00+b01) -> c00 += m1, c11 -= m1
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1)
        add_term(i0*m + j1, j0*p + k1, k0*n + i0, 1) # Wait, U and V need to match the sum
        
        # Strassen steps:
        # m1 = (a00 + a01) * (b00 + b01)
        # U: (i0,j0) and (i0,j1)
        # V: (j0,k0) and (j0,k1)
        # This is NOT a single term. Strassen decomposes the product into 7 terms.
        # We need to emit 7 terms for this block.
        
        # Term 1: (a00+a01) * (b00+b01)
        # U has 1 at (i0,j0) and (i0,j1)
        # V has 1 at (j0,k0) and (j0,k1)
        # W has 1 at (k0,i0) and (k1,i1) -> c00 gets +1, c11 gets -1? 
        # Actually, m1 contributes to c00 and c11.
        # c00 = ... + m1
        # c11 = ... - m1
        # But the term is a single multiplication of two linear forms.
        # The factor matrices U, V, W represent the linear forms.
        # For term t:
        # U_t is a vector, V_t is a vector, W_t is a vector.
        # The product is (U_t . A) * (V_t . B) added to C according to W_t.
        
        # Let's define the 7 Strassen terms explicitly.
        
        # 1. (a00+a01) * (b00+b01)
        # U: a00, a01 -> indices i0,j0 and i0,j1
        # V: b00, b01 -> indices j0,k0 and j0,k1
        # W: c00, c11 -> indices k0,i0 and k1,i1
        # Signs: +1 for c00, -1 for c11
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1)
        add_term(i0*m + j1, j0*p + k1, k0*n + i0, 1) # Wait, add_term adds to the SAME column.
        # I need to add multiple non-zero entries to the SAME column for U, V, W.
        
        # Let's refactor add_term to accept lists or just manually write the columns.
        
        def add_strassen_term(u_entries, v_entries, w_entries):
            nonlocal col
            for idx, val in u_entries:
                U[idx][col] += val
            for idx, val in v_entries:
                V[idx][col] += val
            for idx, val in w_entries:
                W[idx][col] += val
            col += 1

        # 1. m1 = (a00+a01)*(b00+b01)
        add_strassen_term(
            [(i0*m+j0, 1), (i0*m+j1, 1)],
            [(j0*p+k0, 1), (j0*p+k1, 1)],
            [(k0*n+i0, 1), (k1*n+i1, -1)]
        )
        
        # 2. m2 = (a10+a11)*b00
        add_strassen_term(
            [(i1*m+j0, 1), (i1*m+j1, 1)],
            [(j0*p+k0, 1)],
            [(k0*n+i1, 1), (k1*n+i1, -1)]
        )
        
        # 3. m3 = a00*(b01-b11)
        add_strassen_term(
            [(i0*m+j0, 1)],
            [(j0*p+k1, 1), (j1*p+k1, -1)],
            [(k1*n+i0, 1)]
        )
        
        # 4. m4 = a11*(b10-b00)
        add_strassen_term(
            [(i1*m+j1, 1)],
            [(j1*p+k0, 1), (j0*p+k0, -1)],
            [(k0*n+i1, 1)]
        )
        
        # 5. m5 = (a00+a11)*(b00+b11)
        add_strassen_term(
            [(i0*m+j0, 1), (i1*m+j1, 1)],
            [(j0*p+k0, 1), (j1*p+k1, 1)],
            [(k0*n+i0, -1), (k1*n+i0, 1), (k0*n+i1, 1), (k1*n+i1, -1)]
        )
        
        # 6. m6 = (a01-a10)*(b10+b11)
        add_strassen_term(
            [(i0*m+j1, 1), (i1*m+j0, -1)],
            [(j1*p+k0, 1), (j1*p+k1, 1)],
            [(k1*n+i1, 1)]
        )
        
        # 7. m7 = (a01-a11)*(b01+b10)
        add_strassen_term(
            [(i0*m+j1, 1), (i1*m+j1, -1)],
            [(j0*p+k1, 1), (j1*p+k0, 1)],
            [(k0*n+i0, 1)]
        )
        
        return

    # 1. 4x4x4 Block (Indices 0-3)
    # 8 2x2 blocks
    # (0,0)-(1,1) A, (0,0)-(1,1) B -> (0,0)-(1,1) C
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # (0,0)-(1,1) A, (2,2)-(3,3) B -> (0,0)-(1,1) C
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # (2,2)-(3,3) A, (0,0)-(1,1) B -> (0,0)-(1,1) C
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # (2,2)-(3,3) A, (2,2)-(3,3) B -> (0,0)-(1,1) C
    strassen_2x2(2, 3, 2, 3, 0, 1)
    
    # C block (0,0)-(1,1) is done.
    # Now C block (0,0)-(1,1) x (2,2)-(3,3)
    # A (0,0)-(1,1) x (0,0)-(1,1) * B (0,0)-(1,1) x (2,2)-(3,3)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A (0,0)-(1,1) x (2,2)-(3,3) * B (2,2)-(3,3) x (2,2)-(3,3)
    strassen_2x2(0, 1, 2, 3, 2, 3)
    
    # C block (2,2)-(3,3) x (0,0)-(1,1)
    # A (2,2)-(3,3) x (0,0)-(1,1) * B (0,0)-(1,1) x (0,0)-(1,1)
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A (2,2)-(3,3) x (2,2)-(3,3) * B (2,2)-(3,3) x (0,0)-(1,1)
    strassen_2x2(2, 3, 2, 3, 0, 1)
    
    # C block (2,2)-(3,3) x (2,2)-(3,3)
    # A (2,2)-(3,3) x (0,0)-(1,1) * B (0,0)-(1,1) x (2,2)-(3,3)
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A (2,2)-(3,3) x (2,2)-(3,3) * B (2,2)-(3,3) x (2,2)-(3,3)
    strassen_2x2(2, 3, 2, 3, 2, 3)
    
    # Wait, the C blocks are:
    # C00: A00B00 + A01B10
    # C01: A00B01 + A01B11
    # C10: A10B00 + A11B10
    # C11: A10B01 + A11B11
    
    # My calls:
    # 1. A00B00 -> C00 (strassen_2x2(0,1, 0,1, 0,1))
    # 2. A01B10 -> C00 (strassen_2x2(0,1, 2,3, 0,1)) -- A01 is rows 0-1, cols 2-3. B10 is rows 2-3, cols 0-1.
    # 3. A10B00 -> C10 (strassen_2x2(2,3, 0,1, 0,1)) -- A10 is rows 2-3, cols 0-1. B00 is rows 0-1, cols 0-1.
    # 4. A11B10 -> C10 (strassen_2x2(2,3, 2,3, 0,1)) -- A11 is rows 2-3, cols 2-3. B10 is rows 2-3, cols 0-1.
    
    # 5. A00B01 -> C01 (strassen_2x2(0,1, 0,1, 2,3)) -- A00 rows 0-1 cols 0-1. B01 rows 0-1 cols 2-3.
    # 6. A01B11 -> C01 (strassen_2x2(0,1, 2,3, 2,3)) -- A01 rows 0-1 cols 2-3. B11 rows 2-3 cols 2-3.
    
    # 7. A10B01 -> C11 (strassen_2x2(2,3, 0,1, 2,3)) -- A10 rows 2-3 cols 0-1. B01 rows 0-1 cols 2-3.
    # 8. A11B11 -> C11 (strassen_2x2(2,3, 2,3, 2,3)) -- A11 rows 2-3 cols 2-3. B11 rows 2-3 cols 2-3.
    
    # The calls above are correct for the 4x4x4 block. Rank = 8 * 7 = 56.
    
    # 2. A_{0:4, 4} * B_{4, 0:4} -> C_{0:4, 0:4}
    # A is 4x1 (col 4), B is 1x4 (row 4)
    # C_{i,k} = A_{i,4} B_{4,k}
    # Rank 16
    for i in range(4):
        for k in range(4):
            add_term(i*m + 4, 4*p + k, k*n + i, 1)
            
    # 3. A_{0:4, 0:4} * B_{0:4, 4} -> C_{0:4, 4}
    # A is 4x4, B is 4x1
    # C_{i,4} = sum_{j=0}^3 A_{i,j} B_{j,4}
    # Rank 16
    for i in range(4):
        for j in range(4):
            add_term(i*m + j, j*p + 4, 4*n + i, 1)
            
    # 4. A_{0:4, 4} * B_{4, 4} -> C_{0:4, 4}
    # A is 4x1, B is 1x1
    # C_{i,4} = A_{i,4} B_{4,4}
    # Rank 4
    for i in range(4):
        add_term(i*m + 4, 4*p + 4, 4*n + i, 1)
        
    # 5. A_{4, 0:4} * B_{0:4, 0:4} -> C_{4, 0:4}
    # A is 1x4, B is 4x4
    # C_{4,k} = sum_{j=0}^3 A_{4,j} B_{j,k}
    # Rank 16
    for j in range(4):
        for k in range(4):
            add_term(4*m + j, j*p + k, k*n + 4, 1)
            
    # 6. A_{4, 4} * B_{4, 0:4} -> C_{4, 0:4}
    # A is 1x1, B is 1x4
    # C_{4,k} = A_{4,4} B_{4,k}
    # Rank 4
    for k in range(4):
        add_term(4*m + 4, 4*p + k, k*n + 4, 1)
        
    # 7. A_{4, 0:4} * B_{0:4, 4} -> C_{4, 4}
    # A is 1x4, B is 4x1
    # C_{4,4} = sum_{j=0}^3 A_{4,j} B_{j,4}
    # Rank 4
    for j in range(4):
        add_term(4*m + j, j*p + 4, 4*n + 4, 1)
        
    # 8. A_{4, 4} * B_{4, 4} -> C_{4, 4}
    # A is 1x1, B is 1x1
    # C_{4,4} = A_{4,4} B_{4,4}
    # Rank 1
    add_term(4*m + 4, 4*p + 4, 4*n + 4, 1)
    
    assert col == 117, f"Expected 117, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
