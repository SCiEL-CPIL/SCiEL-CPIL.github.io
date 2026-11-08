"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    total_rank = 114
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # 2x2x2 Strassen tensors
    strassen_tensors = [
        {'U': [(0,1),(3,1)], 'V': [(0,1),(3,1)], 'W': [(0,0,1),(1,1,1)]},
        {'U': [(2,1),(3,1)], 'V': [(0,1)], 'W': [(1,0,1),(1,1,-1)]},
        {'U': [(0,1)], 'V': [(1,1),(3,-1)], 'W': [(0,1,1),(0,0,1)]},
        {'U': [(3,1)], 'V': [(2,1),(0,-1)], 'W': [(0,0,1),(1,0,1)]},
        {'U': [(0,1),(1,1)], 'V': [(3,1)], 'W': [(0,1,1),(0,0,-1)]},
        {'U': [(2,1),(0,-1)], 'V': [(0,1),(1,1)], 'W': [(1,1,1)]},
        {'U': [(1,1),(3,-1)], 'V': [(2,1),(3,1)], 'W': [(0,0,1),(0,1,1)]},
    ]
    
    def add_uv_2x2(a_pos, b_pos, offset):
        """Add U and V contributions for 7 Strassen terms. No W."""
        nonlocal col
        for t in range(7):
            c = offset + t
            tensor = strassen_tensors[t]
            for (lr, coeff_u) in tensor['U']:
                for (ar, ac, a_coeff) in a_pos[lr]:
                    U[ar * m + ac][c] += coeff_u * a_coeff
            for (lc, coeff_v) in tensor['V']:
                for (br, bc, b_coeff) in b_pos[lc]:
                    V[br * p + bc][c] += coeff_v * b_coeff
        return offset + 7
    
    def add_w_for_pi(pi_offset, cr, cc, coeff):
        """Add W contributions for Pi to C block at (cr, cc) with coefficient."""
        for t in range(7):
            c = pi_offset + t
            tensor = strassen_tensors[t]
            for (crow, ccol, w_coeff) in tensor['W']:
                W[(cr + crow) * n + (cc + ccol)][c] += coeff * w_coeff
    
    def get_block(br, bc):
        return [
            [(br, bc, 1)],
            [(br, bc+1, 1)],
            [(br+1, bc, 1)],
            [(br+1, bc+1, 1)],
        ]
    
    def combine(blocks):
        result = [[], [], [], []]
        for block in blocks:
            for i in range(4):
                result[i].extend(block[i])
        return result
    
    def negate_block_at(block, row_thresh, col_thresh=None):
        """Negate entries from a specific block."""
        result = [[], [], [], []]
        for i in range(4):
            for (r, c, coeff) in block[i]:
                if row_thresh is not None and r >= row_thresh:
                    result[i].append((r, c, -coeff))
                elif col_thresh is not None and c >= col_thresh:
                    result[i].append((r, c, -coeff))
                else:
                    result[i].append((r, c, coeff))
        return result
    
    # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3)
    # P1 = (A11+A22)(B11+B22)
    col = add_uv_2x2(combine([get_block(0,0), get_block(2,2)]), 
                     combine([get_block(0,0), get_block(2,2)]), col)
    # P2 = (A21+A22)(B11)
    col = add_uv_2x2(combine([get_block(2,0), get_block(2,2)]), 
                     get_block(0,0), col)
    # P3 = (A11)(B12-B22)
    b_p3 = combine([get_block(0,2), get_block(2,2)])
    b_p3 = negate_block_at(b_p3, row_thresh=2)
    col = add_uv_2x2(get_block(0,0), b_p3, col)
    # P4 = (A22)(B21-B11)
    b_p4 = combine([get_block(2,0), get_block(0,0)])
    b_p4 = negate_block_at(b_p4, row_thresh=0, col_thresh=0)  # negate B11 (rows 0-1)
    # Fix: negate only B11
    b_p4 = [[], [], [], []]
    for i in range(4):
        for (r, c, coeff) in combine([get_block(2,0), get_block(0,0)])[i]:
            if r < 2:  # from B11
                b_p4[i].append((r, c, -coeff))
            else:  # from B21
                b_p4[i].append((r, c, coeff))
    col = add_uv_2x2(get_block(2,2), b_p4, col)
    # P5 = (A11+A12)(B22)
    col = add_uv_2x2(combine([get_block(0,0), get_block(0,2)]), 
                     get_block(2,2), col)
    # P6 = (A21-A11)(B11+B12)
    a_p6 = [[], [], [], []]
    for i in range(4):
        for (r, c, coeff) in combine([get_block(2,0), get_block(0,0)])[i]:
            if r < 2:  # from A11
                a_p6[i].append((r, c, -coeff))
            else:  # from A21
                a_p6[i].append((r, c, coeff))
    col = add_uv_2x2(a_p6, combine([get_block(0,0), get_block(0,2)]), col)
    # P7 = (A12-A22)(B21+B22)
    a_p7 = [[], [], [], []]
    for i in range(4):
        for (r, c, coeff) in combine([get_block(0,2), get_block(2,2)])[i]:
            if r >= 2:  # from A22
                a_p7[i].append((r, c, -coeff))
            else:  # from A12
                a_p7[i].append((r, c, coeff))
    col = add_uv_2x2(a_p7, combine([get_block(2,0), get_block(2,2)]), col)
    
    # W for Part 1
    add_w_for_pi(0, 0, 0, 1)   # C11 += P1
    add_w_for_pi(0, 2, 2, 1)   # C22 += P1
    add_w_for_pi(7, 2, 0, 1)   # C21 += P2
    add_w_for_pi(7, 2, 2, -1)  # C22 -= P2
    add_w_for_pi(14, 0, 2, 1)  # C12 += P3
    add_w_for_pi(14, 2, 2, 1)  # C22 += P3
    add_w_for_pi(21, 0, 0, 1)  # C11 += P4
    add_w_for_pi(21, 2, 0, 1)  # C21 += P4
    add_w_for_pi(28, 0, 0, -1) # C11 -= P5
    add_w_for_pi(28, 0, 2, 1)  # C12 += P5
    add_w_for_pi(35, 2, 2, 1)  # C22 += P6
    add_w_for_pi(42, 0, 0, 1)  # C11 += P7
    add_w_for_pi(42, 0, 2, 1)  # C12 += P7
    
    # Part 2: 4x4 * 4x4 (B cols 4-7, C cols 4-7)
    col = 49
    # P1_2 = (A11+A22)(B11_2+B22_2)
    col = add_uv_2x2(combine([get_block(0,0), get_block(2,2)]), 
                     combine([get_block(0,4), get_block(2,6)]), col)
    # P2_2 = (A21+A22)(B11_2)
    col = add_uv_2x2(combine([get_block(2,0), get_block(2,2)]), 
                     get_block(0,4), col)
    # P3_2 = (A11)(B12_2-B22_2)
    b_p3_2 = [[], [], [], []]
    for i in range(4):
        for (r, c, coeff) in combine([get_block(0,6), get_block(2,6)])[i]:
            if c == 6:  # from B22_2
                b_p3_2[i].append((r, c, -coeff))
            else:  # from B12_2
                b_p3_2[i].append((r, c, coeff))
    col = add_uv_2x2(get_block(0,0), b_p3_2, col)
    # P4_2 = (A22)(B21_2-B11_2)
    b_p4_2 = [[], [], [], []]
    for i in range(4):
        for (r, c, coeff) in combine([get_block(2,4), get_block(0,4)])[i]:
            if c == 4:  # from B11_2
                b_p4_2[i].append((r, c, -coeff))
            else:  # from B21_2
                b_p4_2[i].append((r, c, coeff))
    col = add_uv_2x2(get_block(2,2), b_p4_2, col)
    # P5_2 = (A11+A12)(B22_2)
    col = add_uv_2x2(combine([get_block(0,0), get_block(0,2)]), 
                     get_block(2,6), col)
    # P6_2 = (A21-A11)(B11_2+B12_2)
    a_p6_2 = [[], [], [], []]
    for i in range(4):
        for (r, c, coeff) in combine([get_block(2,0), get_block(0,0)])[i]:
            if r < 2:
                a_p6_2[i].append((r, c, -coeff))
            else:
                a_p6_2[i].append((r, c, coeff))
    col = add_uv_2x2(a_p6_2, combine([get_block(0,4), get_block(0,6)]), col)
    # P7_2 = (A12-A22)(B21_2+B22_2)
    a_p7_2 = [[], [], [], []]
    for i in range(4):
        for (r, c, coeff) in combine([get_block(0,2), get_block(2,2)])[i]:
            if r >= 2:
                a_p7_2[i].append((r, c, -coeff))
            else:
                a_p7_2[i].append((r, c, coeff))
    col = add_uv_2x2(a_p7_2, combine([get_block(2,4), get_block(2,6)]), col)
    
    # W for Part 2
    add_w_for_pi(49, 0, 4, 1)
    add_w_for_pi(49, 2, 6, 1)
    add_w_for_pi(56, 2, 4, 1)
    add_w_for_pi(56, 2, 6, -1)
    add_w_for_pi(63, 0, 6, 1)
    add_w_for_pi(63, 2, 6, 1)
    add_w_for_pi(70, 0, 4, 1)
    add_w_for_pi(70, 2, 4, 1)
    add_w_for_pi(77, 0, 4, -1)
    add_w_for_pi(77, 0, 6, 1)
    add_w_for_pi(84, 2, 6, 1)
    add_w_for_pi(91, 0, 4, 1)
    add_w_for_pi(91, 0, 6, 1)
    
    # Part 3: 4x4 * 4x1 (B col 8, C col 8)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
