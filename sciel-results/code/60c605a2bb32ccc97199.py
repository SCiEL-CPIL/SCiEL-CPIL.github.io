import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    # We use a two-level Strassen decomposition.
    # Level 1: Treat 4x4 * 4x6 as (2x2 blocks) * (2x3 blocks)
    #   A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    #   B = [[B00, B01, B02], [B10, B11, B12]] where each Bij is 2x3
    #   C = A*B = [[C00, C01, C02], [C10, C11, C12]] where each Cij is 2x3
    #
    # Level 1 Strassen: 7 terms
    #   M0 = (A00+A11)*(B00+B11)
    #   M1 = (A10+A11)*(B00)
    #   M2 = (A00)*(B01-B11)
    #   M3 = (A11)*(B10-B00)
    #   M4 = (A00+A01)*(B11)
    #   M5 = (A10-A00)*(B00+B01)
    #   M6 = (A01-A11)*(B10+B11)
    #
    # Each M_s is a 2x3 matrix. To compute each 2x3 product, we split
    # into 2x2 * 2x2 (Strassen, 7 mults) + 2x2 * 2x1 (schoolbook, 4 mults) = 11 mults
    #
    # Total rank: 7 * 11 = 77

    R = 77
    U = np.zeros((16, R), dtype=np.int32)
    V = np.zeros((24, R), dtype=np.int32)
    W = np.zeros((24, R), dtype=np.int32)

    # Define the Strassen coefficients for 2x2 blocks
    # For A blocks indexed 0..3: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # For B blocks indexed 0..3: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #
    # alpha[s][a]: coefficient of A-block a in the s-th Strassen term
    # beta[s][b]: coefficient of B-block b in the s-th Strassen term
    # recon[s]: list of (a, b, gamma) meaning output C_ab gets gamma * M_s

    alpha = [
        [1, 0, 0, 1],  # s=0: A00 + A11
        [0, 0, 1, 1],  # s=1: A10 + A11
        [1, 0, 0, 0],  # s=2: A00
        [0, 0, 0, 1],  # s=3: A11
        [1, 1, 0, 0],  # s=4: A00 + A01
        [-1, 0, 1, 0], # s=5: A10 - A00
        [0, 1, 0, -1]  # s=6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # s=0: B00 + B11
        [1, 0, 0, 0],  # s=1: B00
        [0, 1, 0, -1], # s=2: B01 - B11
        [-1, 0, 1, 0], # s=3: B10 - B00
        [0, 0, 0, 1],  # s=4: B11
        [1, 1, 0, 0],  # s=5: B00 + B01
        [0, 0, 1, 1]   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0: C00 += M0, C11 += M0
        [(1, 0, 1), (1, 1, -1)],    # s=1: C10 += M1, C11 -= M1
        [(0, 1, 1), (1, 1, 1)],     # s=2: C01 += M2, C11 += M2
        [(0, 0, 1), (1, 0, 1)],     # s=3: C00 += M3, C10 += M3
        [(0, 0, -1), (0, 1, 1)],    # s=4: C00 -= M4, C01 += M4
        [(1, 1, 1)],                # s=5: C11 += M5
        [(0, 0, 1)]                 # s=6: C00 += M6
    ]

    # Inner Strassen for 2x2x2 (same structure)
    # For a 2x2 matrix X and 2x2 matrix Y, compute Z = X*Y
    # X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # Using same alpha/beta/recon as above (they're universal for 2x2)

    col = 0
    for s in range(7):
        # M_s = (sum_r alpha[s][r] * A_r) * (sum_c beta[s][c] * B_c)
        # where A_r is 2x2, B_c is 2x3
        # Result is 2x3.
        # We compute this as: for cols 0,1 use inner Strassen (2x2*2x2), for col 2 use schoolbook (2x2*2x1)

        # Part 1: Inner Strassen for columns 0,1
        # X_s = sum_r alpha[s][r] * A_r  (2x2)
        # Y_s = sum_c beta[s][c] * B_c[:,0:2]  (2x2)
        # Z_s = X_s * Y_s (2x2) via 7 inner Strassen terms

        for si in range(7):
            # Inner Strassen term si:
            # P_si = (sum_a inner_alpha[si][a] * X_s[.,a_col]) * (sum_b inner_beta[si][b] * Y_s[b_row,.])
            # But X_s and Y_s are already linear combinations, so we combine coefficients.

            # U contributions: for each (a, r) pair
            for a in range(4):
                if alpha[s][a] == 0:
                    continue
                # a indexes position in A-block: row=a//2, col=a%2 within the 2x2 block
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    if alpha[s][r] == 0:
                        continue
                    # r indexes which A-block: ar=r//2, br=r%2
                    ar, br = r // 2, r % 2
                    # Global A index: row = ar*2 + x_row, col = br*2 + x_col
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)

                    # Now for inner Strassen term si, we need inner_alpha[si] applied to X_s
                    # X_s[a_pos] has coefficient alpha[s][a] (already included)
                    # But wait - inner Strassen operates on X_s which is already a combination.
                    # The inner Strassen term si uses:
                    #   P_si = (sum_i2 inner_alpha[si][i2] * X_s[i2]) * (sum_j2 inner_beta[si][j2] * Y_s[j2])
                    # X_s[i2] = sum_r alpha[s][r] * A_r[i2]  -- no, that's not right.
                    #
                    # Let me reconsider. X_s is a 2x2 matrix. Its entries are:
                    # X_s[i,j] = sum_{r: (r//2,r%2)==(i,j)} alpha[s][r] * A_r[i,j]
                    # Wait no. A_r is a 2x2 block. So A_r has entries A_r[i,j] for i,j in {0,1}.
                    # The global A has entries A[ar*2+i, br*2+j] = A_r[i,j] where r = ar*2+br.
                    #
                    # X_s[i,j] = sum_{ar,br: ar*2+br = r} ... 
                    # Actually, for each a in 0..3, a corresponds to a specific (x_row, x_col) position
                    # within a 2x2 block, and we sum over which block r:
                    # X_s[x_row, x_col] = sum_r alpha[s][r] * A_r[x_row, x_col]
                    # = sum_r alpha[s][r] * A[ar*2+x_row, br*2+x_col] where ar=r//2, br=r%2
                    #
                    # So for inner Strassen, X_s is treated as a standard 2x2 matrix.
                    # inner_alpha[si] applies to the entries of X_s:
                    # For inner term si, the U-part coefficient for A[ar*2+x_row, br*2+x_col] is:
                    #   alpha[s][r] * inner_alpha[si][x_row*2+x_col]
                    # where x_row*2+x_col is the position index in X_s.

                    inner_pos = x_row * 2 + x_col
                    coeff = alpha[s][r] * alpha[si][inner_pos]  # Reusing alpha for inner Strassen
                    if coeff != 0:
                        U[u_idx, col] += coeff

            # V contributions: for each (b, c) pair  
            for b in range(4):
                if beta[s][b] == 0:
                    continue
                # b indexes position in B-block (for cols 0,1): row=b//2, col=b%2
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    if beta[s][c] == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    # Global B index: row = ac*2 + y_row, col = bc*3 + y_inner_col
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)

                    inner_pos = y_row * 2 + y_inner_col
                    coeff = beta[s][c] * beta[si][inner_pos]
                    if coeff != 0:
                        V[v_idx, col] += coeff

            # W contributions: output of inner Strassen term si
            # P_si contributes to Z_s[za, zb] with coefficient inner_recon[si]
            # Z_s[za, zb] is the (za, zb) entry of the 2x2 product X_s * Y_s
            # This goes to C_s[za, zb] where C_s is the 2x3 result
            # But C_s[za, zb] is part of the overall C at block (a_out, b_out) position (za, zb)
            # from recon[s].

            for a_out, b_out, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # C block (a_out, b_out) at position (za, zb)
                    # Global C: row = a_out*2 + za, col = b_out*3 + zb
                    # W index: k*n + i where k is col, i is row
                    w_idx = (b_out * 3 + zb) * 4 + (a_out * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Part 2: Schoolbook for column 2
        # X_s (2x2) * Y_s[:,2] (2x1) -> Z_s[:,2] (2x1)
        # Z_s[0,2] = X_s[0,0]*Y_s[0,2] + X_s[0,1]*Y_s[1,2]
        # Z_s[1,2] = X_s[1,0]*Y_s[0,2] + X_s[1,1]*Y_s[1,2]
        # 4 scalar multiplications

        for u in range(2):
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,2]
                
                # U part: X_s[u,v] = sum_r alpha[s][r] * A_r[u,v]
                for r in range(4):
                    if alpha[s][r] == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += alpha[s][r]
                
                # V part: Y_s[v,2] = sum_c beta[s][c] * B_c[v,2]
                for c in range(4):
                    if beta[s][c] == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += beta[s][c]
                
                # W part: contributes to C_s[u,2]
                for a_out, b_out, gamma in recon[s]:
                    # C block (a_out, b_out) at position (u, 2)
                    w_idx = (b_out * 3 + 2) * 4 + (a_out * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1

    # Verify col == R
    assert col == R, f"Column count {col} != expected {R}"

    def to_list(mat):
        return [[int(x) for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
