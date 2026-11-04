"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 10 x 46
    V = [[0] * rank for _ in range(m * p)]  # 25 x 46
    W = [[0] * rank for _ in range(p * n)]  # 10 x 46
    
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        for (i, j), val in u_dict.items():
            U[i * m + j][col] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][col] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][col] = val
        col += 1
    
    def strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add 7 Strassen terms for a 2x2 sub-problem."""
        nonlocal col
        r0, r1 = a_rows
        ac0, ac1 = a_cols
        br0, br1 = b_rows
        bc0, bc1 = b_cols
        cr0, cr1 = c_rows
        cc0, cc1 = c_cols
        
        add_term(
            {(r0, ac0): 1, (r1, ac1): 1},
            {(br0, bc0): 1, (br1, bc1): 1},
            {(cc0, cr0): 1, (cc1, cr1): 1}
        )
        add_term(
            {(r1, ac0): 1, (r1, ac1): 1},
            {(br0, bc0): 1},
            {(cc0, cr1): 1, (cc1, cr1): 1}
        )
        add_term(
            {(r0, ac0): 1},
            {(br0, bc1): 1, (br1, bc1): -1},
            {(cc1, cr0): 1, (cc1, cr1): 1}
        )
        add_term(
            {(r1, ac1): 1},
            {(br1, bc0): 1, (br0, bc0): -1},
            {(cc0, cr0): 1, (cc0, cr1): 1}
        )
        add_term(
            {(r0, ac0): 1, (r0, ac1): 1},
            {(br1, bc1): 1},
            {(cc0, cr0): 1, (cc1, cr0): 1}
        )
        add_term(
            {(r1, ac0): 1, (r0, ac0): -1},
            {(br0, bc0): 1, (br0, bc1): 1},
            {(cc1, cr1): 1}
        )
        add_term(
            {(r0, ac1): 1, (r1, ac1): -1},
            {(br1, bc0): 1, (br1, bc1): 1},
            {(cc0, cr0): 1, (cc1, cr1): 1}
        )
    
    def mul_2x1_by_1x2(a_rows, a_col, b_row, b_cols, c_rows, c_cols):
        """Add 4 terms for a 2x1 by 1x2 sub-problem."""
        nonlocal col
        r0, r1 = a_rows
        ac = a_col
        br = b_row
        bc0, bc1 = b_cols
        cr0, cr1 = c_rows
        cc0, cc1 = c_cols
        
        add_term({(r0, ac): 1}, {(br, bc0): 1}, {(cc0, cr0): 1})
        add_term({(r0, ac): 1}, {(br, bc1): 1}, {(cc1, cr0): 1})
        add_term({(r1, ac): 1}, {(br, bc0): 1}, {(cc0, cr1): 1})
        add_term({(r1, ac): 1}, {(br, bc1): 1}, {(cc1, cr1): 1})
    
    # Column pair (0,1): j-blocks (0,1), (2,3), (4)
    strassen_2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    strassen_2x2([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    mul_2x1_by_1x2([0,1], 4, 4, [0,1], [0,1], [0,1])
    
    # Column pair (2,3): j-blocks (0,1), (2,3), (4)
    strassen_2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    strassen_2x2([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    mul_2x1_by_1x2([0,1], 4, 4, [2,3], [0,1], [2,3])
    
    # Column 4: 10 individual terms
    for j in range(5):
        add_term({(0, j): 1}, {(j, 4): 1}, {(4, 0): 1})
        add_term({(1, j): 1}, {(j, 4): 1}, {(4, 1): 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
