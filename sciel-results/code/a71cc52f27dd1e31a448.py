"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decompose M_{2,3,7} into blocks based on m=2+1 and p=2+2+2+1
    # Blocks:
    # 1. m=2, p=2 -> M_{2,2,2} (Rank 7, Strassen)
    # 2. m=2, p=2 -> M_{2,2,2} (Rank 7, Strassen)
    # 3. m=2, p=2 -> M_{2,2,2} (Rank 7, Strassen)
    # 4. m=2, p=1 -> M_{2,2,1} (Rank 4, Schoolbook)
    # 5. m=1, p=2 -> M_{2,1,2} (Rank 4, Schoolbook)
    # 6. m=1, p=2 -> M_{2,1,2} (Rank 4, Schoolbook)
    # 7. m=1, p=2 -> M_{2,1,2} (Rank 4, Schoolbook)
    # 8. m=1, p=1 -> M_{2,1,1} (Rank 2, Schoolbook)
    
    total_rank = 7 + 7 + 7 + 4 + 4 + 4 + 4 + 2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w, col):
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w

    # Strassen decomposition for M_{2,2,2}
    # A indices: 00, 01, 10, 11 (0,1,2,3)
    # B indices: 00, 10, 01, 11 (0,1,2,3) -> j*2+k
    # C indices: 00, 01, 10, 11 (0,1,2,3) -> i*2+k
    
    def add_strassen_block(m_offset, p_offset):
        nonlocal col_offset
        # A indices: m_offset*2 + 0, m_offset*2 + 1 (rows 0,1 of A block)
        # B indices: (m_offset*2 + 0)*2 + 0, ...
        
        # Local indices in the block
        # A: 0,1,2,3
        # B: 0,1,2,3
        # C: 0,1,2,3
        
        # Strassen terms
        # M1: (A00+A11)(B00+B11)
        add_term(m_offset*2 + 0, (m_offset*2 + 0)*2 + 0, p_offset*2 + 0, 1, 1, 1, col_offset)
        add_term(m_offset*2 + 1, (m_offset*2 + 1)*2 + 1, p_offset*2 + 0, 1, 1, 1, col_offset)
        # Wait, U, V, W are global.
        # U index: i*m + j. For block m_offset, i is global row, j is local col.
        # Global j = m_offset*2 + local_j.
        # Global U index = i*m + (m_offset*2 + local_j).
        # Global V index = (m_offset*2 + local_j)*p + (p_offset*2 + local_k).
        # Global W index = (p_offset*2 + local_k)*n + i.
        
        # Let's redefine add_term to take global indices.
        
        col_offset += 1
        
        # M2: (A10+A11)B00
        add_term(0*2 + 0, 0*2 + 0, 0*2 + 0, 0, 0, 0, 0) # Placeholder, will rewrite below
        
        # I will implement the blocks explicitly in the main loop for clarity and correctness.
        
    # Reset
    col_offset = 0
    
    # Block 1: m=0..1, p=0..1 (M_{2,2,2})
    # Strassen
    # M1
    U[0*3+0][col_offset] = 1; V[0*7+0][col_offset] = 1; W[0*2+0][col_offset] = 1
    U[1*3+1][col_offset] = 1; V[1*7+1][col_offset] = 1; W[0*2+0][col_offset] += 1
    # This is getting complicated with the "add" logic.
    
    # Let's just generate the terms and sum them up in the arrays.
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_strassen(m_off, p_off):
        nonlocal col_offset
        # Local A: 0,1,2,3. Global A: (0*3 + m_off*2+0), (0*3 + m_off*2+1), (1*3 + m_off*2+0), (1*3 + m_off*2+1)
        # Local B: 0,1,2,3. Global B: ((m_off*2+0)*7 + p_off*2+0), ...
        # Local C: 0,1,2,3. Global C: ((p_off*2+0)*2 + 0), ...
        
        a00 = 0*3 + m_off*2 + 0
        a01 = 0*3 + m_off*2 + 1
        a10 = 1*3 + m_off*2 + 0
        a11 = 1*3 + m_off*2 + 1
        
        b00 = (m_off*2+0)*7 + p_off*2 + 0
        b10 = (m_off*2+1)*7 + p_off*2 + 0
        b01 = (m_off*2+0)*7 + p_off*2 + 1
        b11 = (m_off*2+1)*7 + p_off*2 + 1
        
        c00 = (p_off*2+0)*2 + 0
        c01 = (p_off*2+1)*2 + 0
        c10 = (p_off*2+0)*2 + 1
        c11 = (p_off*2+1)*2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[a00][col_offset] += 1
        U[a11][col_offset] += 1
        V[b00][col_offset] += 1
        V[b11][col_offset] += 1
        W[c00][col_offset] += 1
        W[c11][col_offset] += 1
        col_offset += 1
        
        # M2 = (A10+A11)B00
        U[a10][col_offset] += 1
        U[a11][col_offset] += 1
        V[b00][col_offset] += 1
        W[c10][col_offset] += 1
        col_offset += 1
        
        # M3 = A00(B01-B11)
        U[a00][col_offset] += 1
        V[b01][col_offset] += 1
        V[b11][col_offset] -= 1
        W[c01][col_offset] += 1
        col_offset += 1
        
        # M4 = A11(B10-B00)
        U[a11][col_offset] += 1
        V[b10][col_offset] += 1
        V[b00][col_offset] -= 1
        W[c00][col_offset] += 1
        W[c11][col_offset] -= 1
        col_offset += 1
        
        # M5 = (A00+A01)B11
        U[a00][col_offset] += 1
        U[a01][col_offset] += 1
        V[b11][col_offset] += 1
        W[c00][col_offset] -= 1
        W[c01][col_offset] += 1
        col_offset += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10][col_offset] += 1
        U[a00][col_offset] -= 1
        V[b00][col_offset] += 1
        V[b01][col_offset] += 1
        W[c11][col_offset] -= 1
        col_offset += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01][col_offset] += 1
        U[a11][col_offset] -= 1
        V[b10][col_offset] += 1
        V[b11][col_offset] += 1
        W[c00][col_offset] += 1
        W[c11][col_offset] -= 1
        col_offset += 1

    # Add 3 Strassen blocks
    add_strassen(0, 0)
    add_strassen(0, 1)
    add_strassen(0, 2)
    
    # Block 4: m=0..1, p=6 (M_{2,2,1})
    # Schoolbook
    # C00 = A00 B0 + A01 B1
    # C10 = A10 B0 + A11 B1
    
    a00 = 0*3 + 0
    a01 = 0*3 + 1
    a10 = 1*3 + 0
    a11 = 1*3 + 1
    
    b0 = 0*7 + 6
    b1 = 1*7 + 6
    
    c00 = 6*2 + 0
    c10 = 6*2 + 1
    
    # Term 1: A00 B0 -> C00
    U[a00][col_offset] = 1
    V[b0][col_offset] = 1
    W[c00][col_offset] = 1
    col_offset += 1
    
    # Term 2: A01 B1 -> C00
    U[a01][col_offset] = 1
    V[b1][col_offset] = 1
    W[c00][col_offset] = 1
    col_offset += 1
    
    # Term 3: A10 B0 -> C10
    U[a10][col_offset] = 1
    V[b0][col_offset] = 1
    W[c10][col_offset] = 1
    col_offset += 1
    
    # Term 4: A11 B1 -> C10
    U[a11][col_offset] = 1
    V[b1][col_offset] = 1
    W[c10][col_offset] = 1
    col_offset += 1

    # Blocks 5-7: m=2, p=0..1, 2..3, 4..5 (M_{2,1,2})
    # Schoolbook
    # C00 = A0 B0
    # C01 = A0 B1
    # C10 = A1 B0
    # C11 = A1 B1
    
    a0 = 0*3 + 2
    a1 = 1*3 + 2
    
    for p_off in [0, 1, 2]:
        b0 = 2*7 + p_off*2 + 0
        b1 = 2*7 + p_off*2 + 1
        
        c00 = (p_off*2+0)*2 + 0
        c01 = (p_off*2+1)*2 + 0
        c10 = (p_off*2+0)*2 + 1
        c11 = (p_off*2+1)*2 + 1
        
        # Term 1: A0 B0 -> C00
        U[a0][col_offset] = 1
        V[b0][col_offset] = 1
        W[c00][col_offset] = 1
        col_offset += 1
        
        # Term 2: A0 B1 -> C01
        U[a0][col_offset] = 1
        V[b1][col_offset] = 1
        W[c01][col_offset] = 1
        col_offset += 1
        
        # Term 3: A1 B0 -> C10
        U[a1][col_offset] = 1
        V[b0][col_offset] = 1
        W[c10][col_offset] = 1
        col_offset += 1
        
        # Term 4: A1 B1 -> C11
        U[a1][col_offset] = 1
        V[b1][col_offset] = 1
        W[c11][col_offset] = 1
        col_offset += 1

    # Block 8: m=2, p=6 (M_{2,1,1})
    # C00 = A0 B0
    # C10 = A1 B0
    
    b0 = 2*7 + 6
    
    c00 = 6*2 + 0
    c10 = 6*2 + 1
    
    # Term 1: A0 B0 -> C00
    U[a0][col_offset] = 1
    V[b0][col_offset] = 1
    W[c00][col_offset] = 1
    col_offset += 1
    
    # Term 2: A1 B0 -> C10
    U[a1][col_offset] = 1
    V[b0][col_offset] = 1
    W[c10][col_offset] = 1
    col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
