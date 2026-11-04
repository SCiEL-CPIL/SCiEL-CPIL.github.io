"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition of rank 42.
    # Split 2x4x6 into two 2x2x6 multiplications.
    # Each 2x2x6 is split into three 2x2x2 multiplications.
    # Each 2x2x2 uses Strassen (rank 7).
    # Total rank = 2 * 3 * 7 = 42.
    
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for 2x2x2 Strassen:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    strassen_U = [
        [1, 0, 1, 0, 1, 0, 0, 0], # M1: A11+A22
        [0, 0, 0, 0, 0, 0, 0, 0], # dummy
        [0, 0, 0, 0, 0, 0, 0, 0], # dummy
        [0, 0, 0, 0, 0, 0, 0, 0], # dummy
    ]
    # This manual construction is error-prone. Let's use a helper.
    
    def add_strassen_block(offset_u, offset_v, offset_w, base_rank):
        """
        Adds a Strassen decomposition for a 2x2x2 block.
        offset_u: start index in U for this block's A
        offset_v: start index in V for this block's B
        offset_w: start index in W for this block's C
        base_rank: start index in the rank dimension
        """
        # Strassen factors for 2x2x2
        # U rows: 0..3 (A), V rows: 0..3 (B), W rows: 0..3 (C)
        # 7 rank columns
        
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # U vectors (A part)
        u0 = [1, 0, 0, 1] # A00 + A11
        u1 = [0, 0, 1, 1] # A10 + A11
        u2 = [1, 0, 0, 0] # A00
        u3 = [0, 0, 0, 1] # A11
        u4 = [1, 1, 0, 0] # A00 + A01
        u5 = [-1, 0, 1, 0] # A10 - A00
        u6 = [0, 1, 0, -1] # A01 - A11
        
        # V vectors (B part)
        v0 = [1, 0, 0, 1] # B00 + B11
        v1 = [1, 0, 0, 0] # B00
        v2 = [0, 1, 0, -1] # B01 - B11
        v3 = [0, 0, 1, -1] # B10 - B00
        v4 = [0, 0, 0, 1] # B11
        v5 = [1, 1, 0, 0] # B00 + B01
        v6 = [0, 0, 1, 1] # B10 + B11
        
        # W vectors (C part)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        w0 = [1, 0, 0, 1, -1, 0, 1]
        w1 = [0, 0, 1, 0, 1, 0, 0]
        w2 = [0, 1, 0, 1, 0, 0, 0]
        w3 = [1, -1, 1, 0, 0, 1, 0]
        
        for r in range(7):
            col = base_rank + r
            # U
            U[offset_u][col] = u0[r] if r == 0 else (u1[r] if r == 1 else (u2[r] if r == 2 else (u3[r] if r == 3 else (u4[r] if r == 4 else (u5[r] if r == 5 else u6[r])))))
            # V
            V[offset_v][col] = v0[r] if r == 0 else (v1[r] if r == 1 else (v2[r] if r == 2 else (v3[r] if r == 3 else (v4[r] if r == 4 else (v5[r] if r == 5 else v6[r])))))
            # W
            W[offset_w][col] = w0[r] if r == 0 else (w1[r] if r == 1 else (w2[r] if r == 2 else w3[r]))
            
    # Structure:
    # A is 2x4. Split into A1 (2x2) and A2 (2x2).
    # A1 indices in U: 0..3
    # A2 indices in U: 4..7
    # B is 4x6. Split into B1 (2x6) and B2 (2x6).
    # B1 rows 0..1, B2 rows 2..3.
    # B1 cols 0..5. Split into B11 (2x2), B12 (2x2), B13 (2x2).
    # B11 cols 0..1, B12 cols 2..3, B13 cols 4..5.
    # B2 rows 2..3. Split into B21 (2x2), B22 (2x2), B23 (2x2).
    # B21 cols 0..1, B22 cols 2..3, B23 cols 4..5.
    # C is 2x6.
    # C cols 0..5.
    # C11 (from A1 B11) -> C cols 0..1
    # C12 (from A1 B12) -> C cols 2..3
    # C13 (from A1 B13) -> C cols 4..5
    # C21 (from A2 B21) -> C cols 0..1
    # C22 (from A2 B22) -> C cols 2..3
    # C23 (from A2 B23) -> C cols 4..5
    
    # Note: W index is k*n + i.
    # For C11 (cols 0,1):
    # k=0: W[0*2 + 0] = W[0], W[0*2 + 1] = W[1]
    # k=1: W[1*2 + 0] = W[2], W[1*2 + 1] = W[3]
    # So offset_w for C11 is 0.
    
    # Block 1: A1 * B11
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A1 * B12
    # A1 offset 0
    # B12 is B rows 0..1, cols 2..3.
    # V index for B12: row 0 is 0, row 1 is 1. Col 2 is 2, Col 3 is 3.
    # In V (m*p = 24), index is j*p + k.
    # j=0, k=2 -> 0*6+2 = 2.
    # j=0, k=3 -> 3.
    # j=1, k=2 -> 1*6+2 = 8.
    # j=1, k=3 -> 9.
    # So V rows for this block are 2, 3, 8, 9.
    # We need to map the local 2x2 B to these indices.
    # My add_strassen_block assumes contiguous rows 0..3 in V.
    # I need to generalize it to accept a list of V row indices.
    
    # Let's refactor add_strassen_block to accept explicit row mappings.
    
    # Clear U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(u_rows, v_rows, w_rows, base_rank):
        u0 = [1, 0, 0, 1]
        u1 = [0, 0, 1, 1]
        u2 = [1, 0, 0, 0]
        u3 = [0, 0, 0, 1]
        u4 = [1, 1, 0, 0]
        u5 = [-1, 0, 1, 0]
        u6 = [0, 1, 0, -1]
        
        v0 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 0]
        v2 = [0, 1, 0, -1]
        v3 = [0, 0, 1, -1]
        v4 = [0, 0, 0, 1]
        v5 = [1, 1, 0, 0]
        v6 = [0, 0, 1, 1]
        
        w0 = [1, 0, 0, 1, -1, 0, 1]
        w1 = [0, 0, 1, 0, 1, 0, 0]
        w2 = [0, 1, 0, 1, 0, 0, 0]
        w3 = [1, -1, 1, 0, 0, 1, 0]
        
        for r in range(7):
            col = base_rank + r
            for i in range(4):
                U[u_rows[i]][col] += u0[r] if r == 0 else (u1[r] if r == 1 else (u2[r] if r == 2 else (u3[r] if r == 3 else (u4[r] if r == 4 else (u5[r] if r == 5 else u6[r])))))
                V[v_rows[i]][col] += v0[r] if r == 0 else (v1[r] if r == 1 else (v2[r] if r == 2 else (v3[r] if r == 3 else (v4[r] if r == 4 else (v5[r] if r == 5 else v6[r])))))
                W[w_rows[i]][col] += w0[r] if r == 0 else (w1[r] if r == 1 else (w2[r] if r == 2 else w3[r]))

    # Block 1: A1 (rows 0,1 in U) * B11 (rows 0,1; cols 0,1 in V) -> C11 (cols 0,1 in W)
    # U rows: 0, 1, 2, 3
    # V rows: 0*6+0=0, 0*6+1=1, 1*6+0=6, 1*6+1=7
    # W rows: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    add_strassen_block([0, 1, 2, 3], [0, 1, 6, 7], [0, 1, 2, 3], 0)
    
    # Block 2: A1 * B12 (rows 0,1; cols 2,3 in V) -> C12 (cols 2,3 in W)
    # U rows: 0, 1, 2, 3
    # V rows: 0*6+2=2, 0*6+3=3, 1*6+2=8, 1*6+3=9
    # W rows: 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7
    add_strassen_block([0, 1, 2, 3], [2, 3, 8, 9], [4, 5, 6, 7], 7)
    
    # Block 3: A1 * B13 (rows 0,1; cols 4,5 in V) -> C13 (cols 4,5 in W)
    # U rows: 0, 1, 2, 3
    # V rows: 0*6+4=4, 0*6+5=5, 1*6+4=10, 1*6+5=11
    # W rows: 4*2+0=8, 4*2+1=9, 5*2+0=10, 5*2+1=11
    add_strassen_block([0, 1, 2, 3], [4, 5, 10, 11], [8, 9, 10, 11], 14)
    
    # Block 4: A2 (rows 4,5 in U) * B21 (rows 2,3; cols 0,1 in V) -> C21 (cols 0,1 in W)
    # U rows: 4, 5, 6, 7
    # V rows: 2*6+0=12, 2*6+1=13, 3*6+0=18, 3*6+1=19
    # W rows: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    add_strassen_block([4, 5, 6, 7], [12, 13, 18, 19], [0, 1, 2, 3], 21)
    
    # Block 5: A2 * B22 (rows 2,3; cols 2,3 in V) -> C22 (cols 2,3 in W)
    # U rows: 4, 5, 6, 7
    # V rows: 2*6+2=14, 2*6+3=15, 3*6+2=20, 3*6+3=21
    # W rows: 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7
    add_strassen_block([4, 5, 6, 7], [14, 15, 20, 21], [4, 5, 6, 7], 28)
    
    # Block 6: A2 * B23 (rows 2,3; cols 4,5 in V) -> C23 (cols 4,5 in W)
    # U rows: 4, 5, 6, 7
    # V rows: 2*6+4=16, 2*6+5=17, 3*6+4=22, 3*6+5=23
    # W rows: 4*2+0=8, 4*2+1=9, 5*2+0=10, 5*2+1=11
    add_strassen_block([4, 5, 6, 7], [16, 17, 22, 23], [8, 9, 10, 11], 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
