import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """Standard Strassen 2x2x2 with 7 bilinear forms.
    Returns alpha[7][4], beta[7][4], recon[7] where a = i*2+j, b = k*2+l.
    """
    alpha = [
        [1,0,0,1],   # A00 + A11
        [0,0,1,1],   # A10 + A11
        [1,0,0,0],   # A00
        [0,0,0,1],   # A11
        [1,1,0,0],   # A00 + A01
        [-1,0,1,0],  # A10 - A00
        [0,1,0,-1],  # A01 - A11
    ]
    beta = [
        [1,0,0,1],   # B00 + B11
        [1,0,0,0],   # B00
        [0,1,0,-1],  # B01 - B11
        [-1,0,1,0],  # B10 - B00
        [0,0,0,1],   # B11
        [1,1,0,0],   # B00 + B01
        [0,0,1,1],   # B10 + B11
    ]
    recon = [
        [(0,0,1),(1,1,1)],
        [(1,0,1),(1,1,-1)],
        [(0,1,1),(1,1,1)],
        [(0,0,1),(1,0,1)],
        [(0,0,-1),(0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    return alpha, beta, recon


def _build_4x4x4(alpha, beta, recon):
    """Build 4x4x4 Strassen-tiling factor matrices.
    U: (16, 28), V: (24, 28), W: (24, 28)
    B has 6 columns but 444 part uses cols 0..3, so V has 24 rows (4*6) but only cols 0..3 filled.
    Actually for 4x4x4: A is 4x4, B is 4x4, C is 4x4.
    U shape: (n*m, R) = (16, 28)
    V shape: (m*p, R) = (4*6, 28) = (24, 28) -- but only first 4 cols of B used, rows 0..23 for B row 0..3, col 0..3
    W shape: (p*n, R) = (6*4, 28) = (24, 28) -- W[k*n+i], k=0..3 (C cols), i=0..3 (C rows)
    """
    R = 28
    U = np.zeros((16, R))
    V = np.zeros((24, R))  # B is 4x6, only cols 0..3 used
    W = np.zeros((24, R))  # W[k*n+i], k=0..3, i=0..3

    for s in range(7):
        for si in range(7):
            col = s * 7 + si
            # U: A is 4x4. Block A_{r,a} where r=0,1 (row block), a=0,1 (col block)
            # Local index a_local = ar*2 + x_row, col = br*2 + x_col
            # U index = (ar*2 + x_row) * 4 + (br*2 + x_col)
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            # V: B is 4x4 (first 4 cols of 4x6). Block B_{c,b} where c=0,1 (row block), b=0,1 (col block)
            # Local index b_local = ac*2 + y_row, col = bc*2 + y_inner_col
            # V index = (ac*2 + y_row) * 6 + (bc*2 + y_inner_col)  -- B has 6 cols but we use 0..3
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 2 + y_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            # W: C is 4x4. W[k*n+i] = W[col*4 + row]
            # C block C_{b,a} where b=0,1 (col block), a=0,1 (row block)
            # recon[s] gives (a_block, b_block, gamma)
            # recon[si] gives (za, zb, zgamma)
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # C row = a*2 + za, C col = b*2 + zb
                    w_idx = (b * 2 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

    return U, V, W


def _build_4x4x2(alpha, beta, recon):
    """Build 4x4x2 factor matrices for multiplying 4x4 by 4x2.
    Uses 2 Strassen 2x2x2 blocks (top and bottom rows of C).
    U: (16, 14), V: (24, 14), W: (24, 14)
    """
    R = 14
    U = np.zeros((16, R))
    V = np.zeros((24, R))  # B is 4x6, cols 4..5 used
    W = np.zeros((24, R))  # W[k*n+i], k=4,5 (C cols), i=0..3

    # Two row-blocks of C: rows 0,1 (top) and rows 2,3 (bottom)
    # For each row-block, we do a Strassen 2x2 (A block) x 2x2 (B block) x 2x2 (C block)
    # But C is 2x2 per block: rows (2*rb+0, 2*rb+1), cols (4,5)
    
    for rb in range(2):  # row block of C
        for s in range(7):
            for si in range(7):
                col = rb * 7 + s * 7 + si  # 0..13
                # U: A row-block is (rb*2, rb*2+1), all 4 cols
                for a in range(4):
                    if alpha[si][a] != 0:
                        x_row, x_col = a // 2, a % 2
                        for r in range(4):
                            if alpha[s][r] != 0:
                                ar, br = r // 2, r % 2
                                u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                                U[u_idx, col] += alpha[si][a] * alpha[s][r]
                # V: B is 4x2 (cols 4,5). B block: rows (ac*2+y_row), cols (bc*2+y_col) but B only has 2 cols
                # B global col = 4 + (bc*2 + y_col) but B is 4x2, so cols 0,1 locally -> global 4,5
                # Actually B is 4x6. For 4x4x2, B uses cols 4,5.
                # B block: row block ac (0,1), col is just 0 or 1 (2 cols total)
                # V index = row * 6 + col
                for b in range(4):
                    if beta[si][b] != 0:
                        y_row, y_col = b // 2, b % 2
                        for c in range(4):
                            if beta[s][c] != 0:
                                ac, bc = c // 2, c % 2
                                v_idx = (ac * 2 + y_row) * 6 + (4 + bc * 2 + y_col)
                                V[v_idx, col] += beta[si][b] * beta[s][c]
                # W: C is 4x2. W[k*n+i], k=4,5, i=0..3
                # C block: rows (rb*2 + a*2 + za), cols (4 + b*2 + zb) but C is 4x2 so cols are 4,5
                # recon[s]: (a_block, b_block, gamma) where a_block is row-block of C, b_block is col-block
                # But C is only 2 cols, so b_block only 0? No, Strassen 2x2x2 needs 2x2 C.
                # C is 2 rows x 2 cols per block. Rows: rb*2 + a*2 + za, Cols: 4 + b*2 + zb? No, C is 4x2.
                # C cols are 4,5. So within a 2x2 C block, cols are 4 and 5 (b=0 -> col 4, b=1 -> col 5)?
                # Actually Strassen 2x2x2: C is 2x2. Here C block is 2 rows x 2 cols.
                # C row = rb*2 + a*2 + za, C col = 4 + b*2 + zb? But C only has cols 4,5.
                # b*2 + zb: b=0 -> 0,1; b=1 -> 2,3. But C cols are 4,5. So b maps to col offset 0 or 1?
                # No: Strassen 2x2x2 assumes 2x2 matrices. C is 2x2 here: rows (rb*2, rb*2+1) from A side, cols from B side.
                # Wait, I'm confusing myself. Let me redo.
                
    # Let me restart the 4x4x2 construction more carefully.
    # A is 4x4, B is 4x2, C is 4x2.
    # Split A into 2x2 blocks: A_{I,J} for I,J in {0,1}. I=row block, J=col block.
    # Split B into 2x2 blocks vertically: B_{I} for I in {0,1} (row blocks), each is 2x2 (2 rows, 2 cols).
    # Split C into 2x2 blocks: C_{I} for I in {0,1} (row blocks), each is 2x2.
    
    # For C row-block I: C_I = A_0 * B_0 + A_1 * B_1 (2x2 = 2x2 * 2x2 + 2x2 * 2x2)
    # Each 2x2*2x2 product uses Strassen rank 7.
    # So C_0 uses 7+7=14 forms? No, C_0 = A_0*B_0 + A_1*B_1, each 2x2*2x2 is rank 7.
    # So total for C_0: 14, C_1: 14, total 28. That's too much.
    
    # Alternative: C = A * B where A is 4x4, B is 4x2.
    # C row-block I: C_I = A_{I,0} * B_0 + A_{I,1} * B_1 where A_{I,J} is 2x2, B_J is 2x2.
    # So C_0 = A_{0,0}*B_0 + A_{0,1}*B_1, C_1 = A_{1,0}*B_0 + A_{1,1}*B_1.
    # Each term is 2x2*2x2 = rank 7. Total: 4*7 = 28. Still too much.
    
    # Better: Use Strassen on the 4x4 x 4x4 part for cols 0..3, and for cols 4..5,
    # note that C[:,4:6] = A * B[:,4:6] where B[:,4:6] is 4x2.
    # We can write this as: for each row-block of C (2 rows), 
    # C_{I,0:2} (2x2) = A_{I,0} * B_{0,4:6} + A_{I,1} * B_{1,4:6}
    # where A_{I,J} is 2x2, B_{I,4:6} is 2x2.
    # That's 2x2*2x2 + 2x2*2x2 = 7+7 = 14 per row-block, 28 total.
    
    # Hmm, that's rank 28 for the 4x4x2 part. Total would be 28+28=56.
    # But actually, we can do better: use the fact that B[:,4:6] is 4x2.
    # C[:,4:6] = A * B[:,4:6]. This is a 4x4 times 4x2.
    # We can tile: split A into 2x2 blocks (4 blocks), B[:,4:6] into 2 row-blocks of 2x2.
    # C row-block I (2 rows, 2 cols) = sum_J A_{I,J} * B_{J,4:6} (J=0,1)
    # Each A_{I,J} * B_{J,4:6} is 2x2 * 2x2 = rank 7.
    # So C row-block I uses 2*7 = 14 forms. Total 28.
    
    # So the 4x4x2 part has rank 28. Total rank = 28 + 28 = 56.
    
    # But wait - can we do better for 4x4x2? 
    # 4x4x2: A(4x4) * B(4x2) = C(4x2)
    # Split A into 2x2 blocks: A = [[A00,A01],[A10,A11]]
    # B into 2 row-blocks: B = [[B0],[B1]] where B0, B1 are 2x2
    # C = [[A00*B0 + A01*B1], [A10*B0 + A11*B1]]
    # C_top (2x2) = A00*B0 + A01*B1
    # C_bot (2x2) = A10*B0 + A11*B1
    # Each 2x2*2x2 is rank 7. Total: 4*7 = 28.
    # So rank 28 for 4x4x2.
    
    # Total: 28 (4x4x4) + 28 (4x4x2) = 56.
    
    # Let me build it properly.
    U2 = np.zeros((16, 28))
    V2 = np.zeros((24, 28))
    W2 = np.zeros((24, 28))
    
    # C_top (rows 0,1; cols 4,5) = A00*B0 + A01*B1
    # C_bot (rows 2,3; cols 4,5) = A10*B0 + A11*B1
    # A00: rows 0,1; cols 0,1
    # A01: rows 0,1; cols 2,3
    # A10: rows 2,3; cols 0,1
    # A11: rows 2,3; cols 2,3
    # B0: rows 0,1; cols 4,5
    # B1: rows 2,3; cols 4,5
    
    terms = [
        # (A_block_row_offset, A_block_col_offset, B_block_row_offset, C_row_offset, C_col_offset)
        (0, 0, 0, 0, 4),  # A00 * B0 -> C rows 0,1; cols 4,5
        (0, 2, 0, 0, 4),  # A01 * B1 -> C rows 0,1; cols 4,5  (A01 is rows 0,1 cols 2,3; B1 is rows 2,3)
        (2, 0, 0, 2, 4),  # A10 * B0 -> C rows 2,3; cols 4,5
        (2, 2, 0, 2, 4),  # A11 * B1 -> C rows 2,3; cols 4,5
    ]
    # Wait, B0 is rows 0,1 cols 4,5. B1 is rows 2,3 cols 4,5.
    # A00*B0: A00 (2x2) * B0 (2x2) = 2x2. A00 rows 0,1 cols 0,1. B0 rows 0,1 cols 4,5.
    # A01*B1: A01 (2x2) * B1 (2x2) = 2x2. A01 rows 0,1 cols 2,3. B1 rows 2,3 cols 4,5.
    # This is correct.
    
    col = 0
    for (ar_off, ac_off, br_off, cr_off, cc_off) in terms:
        for s in range(7):
            for si in range(7):
                c = col
                # U: A block at (ar_off, ac_off), 2x2
                for a in range(4):
                    if alpha[si][a] != 0:
                        x_row, x_col = a // 2, a % 2
                        for r in range(4):
                            if alpha[s][r] != 0:
                                ar, br = r // 2, r % 2
                                u_idx = (ar_off + ar*2 + x_row) * 4 + (ac_off + br*2 + x_col)
                                U2[u_idx, c] += alpha[si][a] * alpha[s][r]
                # V: B block at (br_off, cc_off), 2x2. B global row = br_off + ac*2 + y_row, col = cc_off + bc*2 + y_col
                for b in range(4):
                    if beta[si][b] != 0:
                        y_row, y_col = b // 2, b % 2
                        for c2 in range(4):
                            if beta[s][c2] != 0:
                                ac2, bc2 = c2 // 2, c2 % 2
                                v_idx = (br_off + ac2*2 + y_row) * 6 + (cc_off + bc2*2 + y_col)
                                V2[v_idx, c] += beta[si][b] * beta[s][c2]
                # W: C block at (cr_off, cc_off), 2x2. W[k*n+i] = W[col*4 + row]
                for a, b, gamma in recon[s]:
                    for za, zb, zgamma in recon[si]:
                        w_row = cr_off + a*2 + za
                        w_col = cc_off + b*2 + zb
                        w_idx = w_col * 4 + w_row
                        W2[w_idx, c] += gamma * zgamma
                col += 1
    
    return U2, V2, W2


def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    alpha, beta, recon = _strassen_2x2()
    
    # 4x4x4 part (cols 0..3)
    U1, V1, W1 = _build_4x4x4(alpha, beta, recon)
    # 4x4x2 part (cols 4..5)
    U2, V2, W2 = _build_4x4x2(alpha, beta, recon)
    
    # Concatenate
    U = np.hstack([U1, U2])  # (16, 56)
    V = np.hstack([V1, V2])  # (24, 56)
    W = np.hstack([W1, W2])  # (24, 56)
    
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for x in row:
                xr = round(x)
                if abs(x - xr) < 1e-9:
                    r.append(int(xr))
                else:
                    r.append(x)
            result.append(r)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
