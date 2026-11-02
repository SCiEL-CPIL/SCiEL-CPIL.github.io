"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col_idx
        for i, val in enumerate(u_vec):
            if val != 0:
                U[i][col_idx] += val
        for i, val in enumerate(v_vec):
            if val != 0:
                V[i][col_idx] += val
        for i, val in enumerate(w_vec):
            if val != 0:
                W[i][col_idx] += val
        col_idx += 1

    # A indices:
    # a00=0, a01=1, a02=2
    # a10=3, a11=4, a12=5
    
    # B indices:
    # b00=0, b01=1, b02=2, b03=3, b04=4, b05=5, b06=6
    # b10=7, b11=8, b12=9, b13=10, b14=11, b15=12, b16=13
    # b20=14, b21=15, b22=16, b23=17, b24=18, b25=19, b26=20
    
    # W indices:
    # c00=0, c10=1
    # c01=2, c11=3
    # c02=4, c12=5
    # c03=6, c13=7
    # c04=8, c14=9
    # c05=10, c15=11
    # c06=12, c16=13
    
    # Strassen for 2x2x2 blocks
    # Inputs: a00, a01, a10, a11 and b_00, b_01, b_10, b_11
    # Outputs: c_00, c_01, c_10, c_11 (local)
    
    def strassen_block(b00, b01, b10, b11, w_base):
        # w_base is the index of c_00 in W
        # c_00 -> w_base
        # c_01 -> w_base + 2
        # c_10 -> w_base + 1
        # c_11 -> w_base + 3
        
        # M1 = (a00+a11)(b00+b11)
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0]=1; u[4]=1
        v[b00]=1; v[b11]=1
        w[w_base]=1; w[w_base+2]=-1; w[w_base+1]=-1; w[w_base+3]=1
        # Wait, C11 = M1 + M4 - M5 + M7. C12 = M3 + M5. C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
        # Let's map carefully.
        # Local C00 (row 0, col 0 of 2x2 block) -> Global W index w_base
        # Local C01 (row 0, col 1) -> Global W index w_base + 2
        # Local C10 (row 1, col 0) -> Global W index w_base + 1
        # Local C11 (row 1, col 1) -> Global W index w_base + 3
        
        # M1 contributes to:
        # C00: +1
        # C11: +1
        # C01: 0
        # C10: 0
        # So w[w_base]=1, w[w_base+3]=1.
        
        # Let's re-verify Strassen formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1:
        # C11 += 1
        # C22 += 1
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0]=1; u[4]=1
        v[b00]=1; v[b11]=1
        w[w_base]=1; w[w_base+3]=1
        add_term(u, v, w)
        
        # M2 = (a10+a11)b00
        # C21 += 1
        # C22 -= 1
        u = [0]*6; v = [0]*21; w = [0]*14
        u[3]=1; u[4]=1
        v[b00]=1
        w[w_base+1]=1; w[w_base+3]=-1
        add_term(u, v, w)
        
        # M3 = a00(b01-b11)
        # C12 += 1
        # C22 += 1
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0]=1
        v[b01]=1; v[b11]=-1
        w[w_base+2]=1; w[w_base+3]=1
        add_term(u, v, w)
        
        # M4 = a11(b10-b00)
        # C11 += 1
        # C21 += 1
        u = [0]*6; v = [0]*21; w = [0]*14
        u[4]=1
        v[b10]=1; v[b00]=-1
        w[w_base]=1; w[w_base+1]=1
        add_term(u, v, w)
        
        # M5 = (a00+a01)b11
        # C11 -= 1
        # C12 += 1
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0]=1; u[1]=1
        v[b11]=1
        w[w_base]=-1; w[w_base+2]=1
        add_term(u, v, w)
        
        # M6 = (a10-a00)(b00+b01)
        # C22 += 1
        u = [0]*6; v = [0]*21; w = [0]*14
        u[3]=1; u[0]=-1
        v[b00]=1; v[b01]=1
        w[w_base+3]=1
        add_term(u, v, w)
        
        # M7 = (a01-a11)(b10+b11)
        # C11 += 1
        u = [0]*6; v = [0]*21; w = [0]*14
        u[1]=1; u[4]=-1
        v[b10]=1; v[b11]=1
        w[w_base]=1
        add_term(u, v, w)

    # Schoolbook for 2x2x1 block
    def schoolbook_block(b00, b10, w_base):
        # b00 is top, b10 is bottom
        # C00 = a00*b00 + a01*b00
        # C10 = a10*b10 + a11*b10
        
        # N1 = a00*b00 -> C00
        u = [0]*6; v = [0]*21; w = [0]*14
        u[0]=1
        v[b00]=1
        w[w_base]=1
        add_term(u, v, w)
        
        # N2 = a01*b00 -> C00
        u = [0]*6; v = [0]*21; w = [0]*14
        u[1]=1
        v[b00]=1
        w[w_base]=1
        add_term(u, v, w)
        
        # N3 = a10*b10 -> C10
        u = [0]*6; v = [0]*21; w = [0]*14
        u[3]=1
        v[b10]=1
        w[w_base+1]=1
        add_term(u, v, w)
        
        # N4 = a11*b10 -> C10
        u = [0]*6; v = [0]*21; w = [0]*14
        u[4]=1
        v[b10]=1
        w[w_base+1]=1
        add_term(u, v, w)

    # Apply Strassen to Group 1 (Cols 0, 1)
    # B indices: b00=0, b01=1, b10=7, b11=8
    # W base: Col 0 -> 0
    strassen_block(0, 1, 7, 8, 0)
    
    # Apply Strassen to Group 2 (Cols 2, 3)
    # B indices: b02=2, b03=3, b12=9, b13=10
    # W base: Col 2 -> 4
    strassen_block(2, 3, 9, 10, 4)
    
    # Apply Strassen to Group 3 (Cols 4, 5)
    # B indices: b04=4, b05=5, b14=11, b15=12
    # W base: Col 4 -> 8
    strassen_block(4, 5, 11, 12, 8)
    
    # Apply Schoolbook to Group 4 (Col 6)
    # B indices: b06=6, b16=13
    # W base: Col 6 -> 12
    schoolbook_block(6, 13, 12)
    
    # Part 2: a3 b3
    # a3 indices: a02=2, a12=5
    # b3 indices: b20=14 .. b26=20
    # W indices: Col j, Row i -> j*2 + i
    
    for j in range(7):
        b_idx = 14 + j
        w_idx_0 = j * 2 + 0
        w_idx_1 = j * 2 + 1
        
        # P0j = a02 * b2j -> W[w_idx_0]
        u = [0]*6; v = [0]*21; w = [0]*14
        u[2]=1
        v[b_idx]=1
        w[w_idx_0]=1
        add_term(u, v, w)
        
        # P1j = a12 * b2j -> W[w_idx_1]
        u = [0]*6; v = [0]*21; w = [0]*14
        u[5]=1
        v[b_idx]=1
        w[w_idx_1]=1
        add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
