"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    
    all_columns = []
    
    def add_column(u_dict, v_dict, w_dict):
        all_columns.append((u_dict, v_dict, w_dict))
    
    def add_strassen(a_base, b_base, c_base):
        a00, a01, a10, a11 = a_base+0, a_base+1, a_base+2, a_base+3
        b00, b01, b10, b11 = b_base+0, b_base+1, b_base+2, b_base+3
        c00, c01, c10, c11 = c_base+0, c_base+1, c_base+2, c_base+3
        
        add_column({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        add_column({a10: 1, a11: 1}, {b00: 1}, {c10: 1})
        add_column({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: -1})
        add_column({a11: 1}, {b10: 1, b00: -1}, {c10: 1, c11: -1})
        add_column({a00: 1, a01: 1}, {b11: 1}, {c01: 1, c11: 1})
        add_column({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c10: 1, c00: -1})
        add_column({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c11: 1, c01: -1})
    
    def add_schoolbook_2x2x2xp(a_base, b_base, c_base, p_sub):
        for i in range(2):
            for j in range(2):
                for k in range(p_sub):
                    a_idx = a_base + i*2 + j
                    b_idx = b_base + j*p_sub + k
                    c_idx = c_base + k*2 + i
                    add_column({a_idx: 1}, {b_idx: 1}, {c_idx: 1})
    
    # C[0:2,:] = A00×B0 + A01×B1
    add_schoolbook_2x2x2xp(0, 0, 0, 3)
    add_strassen(0, 7, 12)
    add_strassen(0, 14, 20)
    
    add_schoolbook_2x2x2xp(2, 0, 0, 3)
    add_strassen(2, 7, 12)
    add_strassen(2, 14, 20)
    
    # C[2:4,:] = A10×B0 + A11×B1
    add_schoolbook_2x2x2xp(8, 0, 2, 3)
    add_strassen(8, 7, 14)
    add_strassen(8, 14, 22)
    
    add_schoolbook_2x2x2xp(10, 0, 2, 3)
    add_strassen(10, 7, 14)
    add_strassen(10, 14, 22)
    
    rank = len(all_columns)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(all_columns):
        for idx, coeff in u_dict.items():
            U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
