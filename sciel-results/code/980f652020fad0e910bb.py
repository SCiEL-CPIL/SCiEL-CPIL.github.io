import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen ---
    
    # Block Strassen:
    # M1: X=A00+A11, Y=B00+B11
    # M2: X=A10+A11, Y=B00
    # M3: X=A00, Y=B10-B11
    # M4: X=A11, Y=B01-B00
    # M5: X=A00+A01, Y=B11
    # M6: X=A10-A00, Y=B00+B10
    # M7: X=A01-A11, Y=B10+B11
    
    # Coefficients for X (A blocks) and Y (B blocks)
    # Index: [k][block_row][block_col]
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    block_x_coeffs = [
        # M1
        [[1, 1], [0, 0], [0, 0], [1, 1]], # A00(1), A11(1) -> Wait, A00 is (0,0), A11 is (1,1)
        # Let's use a dict or 2D array [block_r][block_c]
        # M1: A00 + A11
        [[1, 0], [0, 1]], # A00, A11
        # M2: A10 + A11
        [[0, 0], [1, 1]],
        # M3: A00
        [[1, 0], [0, 0]],
        # M4: A11
        [[0, 0], [0, 1]],
        # M5: A00 + A01
        [[1, 1], [0, 0]],
        # M6: A10 - A00
        [[-1, 0], [1, 0]],
        # M7: A01 - A11
        [[0, 1], [0, -1]],
    ]
    
    block_y_coeffs = [
        # M1: B00 + B11
        [[1, 0], [0, 1]],
        # M2: B00
        [[1, 0], [0, 0]],
        # M3: B10 - B11
        [[0, 0], [1, -1]],
        # M4: B01 - B00
        [[-1, 1], [0, 0]],
        # M5: B11
        [[0, 0], [0, 1]],
        # M6: B00 + B10
        [[1, 0], [1, 0]],
        # M7: B10 + B11
        [[0, 0], [1, 1]],
    ]
    
    # Block Strassen Output Coefficients
    # C00 = M1 - M2 + M6
    # C01 = M2 + M4
    # C10 = M3 + M7
    # C11 = M1 - M4 + M5 - M7
    
    # gamma[RC][k][pq]
    # RC: 00, 01, 10, 11
    # k: 1..7
    # pq: 00, 01, 10, 11
    
    # We need to map (k, pq) to (RC, coeff)
    # Actually, C_RC = sum_k sum_pq gamma[RC][k][pq] M_k[pq]
    
    # M1: C00(1), C11(1)
    # M2: C00(-1), C01(1)
    # M3: C10(1)
    # M4: C01(1), C11(-1)
    # M5: C11(1)
    # M6: C00(1)
    # M7: C10(1), C11(-1)
    
    # Let's build a list of contributions for each k, pq
    # contrib[k][pq] = list of (RC, coeff)
    
    contrib = [[[] for _ in range(4)] for _ in range(7)]
    
    # M1 (k=0)
    contrib[0][0].append((0, 1)) # C00
    contrib[0][3].append((3, 1)) # C11
    
    # M2 (k=1)
    contrib[1][0].append((0, -1)) # C00
    contrib[1][1].append((1, 1))  # C01
    
    # M3 (k=2)
    contrib[2][0].append((2, 1))  # C10
    
    # M4 (k=3)
    contrib[3][1].append((1, 1))  # C01
    contrib[3][3].append((3, -1)) # C11
    
    # M5 (k=4)
    contrib[4][3].append((3, 1))  # C11
    
    # M6 (k=5)
    contrib[5][0].append((0, 1))  # C00
    
    # M7 (k=6)
    contrib[6][0].append((2, 1))  # C10
    contrib[6][3].append((3, -1)) # C11

    # Scalar Strassen (2x2x2)
    # Terms t=0..6
    # u_t, v_t coefficients for X, Y
    # beta_t[pq] coefficients for Z
    
    # t1: (X00+X11)(Y00+Y11) -> Z00(1), Z11(1)
    # t2: (X10+X11)Y00 -> Z00(-1), Z01(1)
    # t3: X00(Y10-Y11) -> Z10(1)
    # t4: X11(Y01-Y00) -> Z01(1), Z11(-1)
    # t5: (X00+X01)Y11 -> Z11(1)
    # t6: (X10-X00)(Y00+Y10) -> Z00(1)
    # t7: (X01-X11)(Y10+Y11) -> Z10(1), Z11(-1)
    
    # Wait, standard Strassen:
    # Z00 = M1 - M2 + M6
    # Z01 = M2 + M4
    # Z10 = M3 + M7
    # Z11 = M1 - M4 + M5 - M7
    
    # So:
    # M1: Z00(1), Z11(1)
    # M2: Z00(-1), Z01(1)
    # M3: Z10(1)
    # M4: Z01(1), Z11(-1)
    # M5: Z11(1)
    # M6: Z00(1)
    # M7: Z10(1), Z11(-1)
    
    scalar_contrib = [[[] for _ in range(4)] for _ in range(7)]
    
    # t=0 (M1)
    scalar_contrib[0][0].append((0, 1))
    scalar_contrib[0][3].append((3, 1))
    
    # t=1 (M2)
    scalar_contrib[1][0].append((0, -1))
    scalar_contrib[1][1].append((1, 1))
    
    # t=2 (M3)
    scalar_contrib[2][0].append((2, 1))
    
    # t=3 (M4)
    scalar_contrib[3][1].append((1, 1))
    scalar_contrib[3][3].append((3, -1))
    
    # t=4 (M5)
    scalar_contrib[4][3].append((3, 1))
    
    # t=5 (M6)
    scalar_contrib[5][0].append((0, 1))
    
    # t=6 (M7)
    scalar_contrib[6][0].append((2, 1))
    scalar_contrib[6][3].append((3, -1))
    
    # Linear forms for scalar Strassen
    # u_t[2][2], v_t[2][2]
    
    # t1: X00+X11, Y00+Y11
    u0 = [[1, 0], [0, 1]]
    v0 = [[1, 0], [0, 1]]
    
    # t2: X10+X11, Y00
    u1 = [[0, 0], [1, 1]]
    v1 = [[1, 0], [0, 0]]
    
    # t3: X00, Y10-Y11
    u2 = [[1, 0], [0, 0]]
    v2 = [[0, 0], [1, -1]]
    
    # t4: X11, Y01-Y00
    u3 = [[0, 0], [0, 1]]
    v3 = [[-1, 1], [0, 0]]
    
    # t5: X00+X01, Y11
    u4 = [[1, 1], [0, 0]]
    v4 = [[0, 0], [0, 1]]
    
    # t6: X10-X00, Y00+Y10
    u5 = [[-1, 0], [1, 0]]
    v5 = [[1, 0], [1, 0]]
    
    # t7: X01-X11, Y10+Y11
    u6 = [[0, 1], [0, -1]]
    v6 = [[0, 0], [1, 1]]
    
    u_forms = [u0, u1, u2, u3, u4, u5, u6]
    v_forms = [v0, v1, v2, v3, v4, v5, v6]
    
    col = 0
    for k in range(7):
        for t in range(7):
            # Construct U, V, W for this term
            
            # U: coeffs of A
            # L_X = sum_{i,j} u_t[i][j] X_k[i][j]
            # X_k[i][j] = sum_{b,c} block_x_coeffs[k][b][c] A_{2b+i, 2c+j}
            
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for i in range(2):
                for j in range(2):
                    coeff_u = u_forms[t][i][j]
                    if coeff_u == 0: continue
                    for b in range(2):
                        for c in range(2):
                            coeff_block = block_x_coeffs[k][b][c]
                            if coeff_block == 0: continue
                            total_coeff = coeff_u * coeff_block
                            if total_coeff == 0: continue
                            a_idx = (2*b + i) * m + (2*c + j)
                            u_vec[a_idx] += total_coeff
                            
            for i in range(2):
                for j in range(2):
                    coeff_v = v_forms[t][i][j]
                    if coeff_v == 0: continue
                    for b in range(2):
                        for c in range(2):
                            coeff_block = block_y_coeffs[k][b][c]
                            if coeff_block == 0: continue
                            total_coeff = coeff_v * coeff_block
                            if total_coeff == 0: continue
                            b_idx = (2*b + i) * p + (2*c + j)
                            v_vec[b_idx] += total_coeff
                            
            # W: coeffs of C
            # S_kt contributes to M_k[pq] with coeff beta_t[pq]
            # M_k[pq] contributes to C_RC with coeff gamma_k_RC_pq
            
            for pq in range(4):
                # Get scalar contrib for this t, pq
                # scalar_contrib[t][pq] is list of (RC, coeff)? No, it's (Z_idx, coeff)
                # Wait, my scalar_contrib structure:
                # scalar_contrib[t][pq] = list of (RC, coeff) is WRONG.
                # It should be: for a given t, what is the coeff of Z_pq?
                # My previous definition:
                # scalar_contrib[t][pq].append((RC, coeff)) was for Block Strassen.
                # For Scalar Strassen, I need:
                # Z_pq = sum_t beta_t_pq S_t
                # So for a fixed t, S_t contributes to Z_pq with coeff beta_t_pq.
                
                # Let's re-define scalar_beta[t][pq]
                # t=0: Z00(1), Z11(1)
                # t=1: Z00(-1), Z01(1)
                # ...
                
                pass # Will implement in code

            # Actually, let's just hardcode the beta matrix
            # beta[t][pq]
            
            # t=0: [1, 0, 0, 1]
            # t=1: [-1, 1, 0, 0]
            # t=2: [0, 0, 1, 0]
            # t=3: [0, 1, 0, -1]
            # t=4: [0, 0, 0, 1]
            # t=5: [1, 0, 0, 0]
            # t=6: [0, 0, 1, -1]
            
            beta = [
                [1, 0, 0, 1],
                [-1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, -1],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, -1]
            ]
            
            # Block Gamma
            # gamma[k][RC][pq]
            # k=0 (M1): C00(1), C11(1) -> RC0: pq0(1), RC3: pq3(1)
            # k=1 (M2): C00(-1), C01(1) -> RC0: pq0(-1), RC1: pq1(1)
            # k=2 (M3): C10(1) -> RC2: pq0(1)
            # k=3 (M4): C01(1), C11(-1) -> RC1: pq1(1), RC3: pq3(-1)
            # k=4 (M5): C11(1) -> RC3: pq3(1)
            # k=5 (M6): C00(1) -> RC0: pq0(1)
            # k=6 (M7): C10(1), C11(-1) -> RC2: pq0(1), RC3: pq3(-1)
            
            # We need to sum over pq
            # W_vec[C_idx] += sum_pq beta[t][pq] * gamma[k][RC][pq]
            
            # Let's build gamma matrix
            # gamma[k][RC][pq]
            
            # Initialize 0
            gamma = [[[0]*4 for _ in range(4)] for _ in range(7)]
            
            # k=0
            gamma[0][0][0] = 1
            gamma[0][3][3] = 1
            # k=1
            gamma[1][0][0] = -1
            gamma[1][1][1] = 1
            # k=2
            gamma[2][2][0] = 1
            # k=3
            gamma[3][1][1] = 1
            gamma[3][3][3] = -1
            # k=4
            gamma[4][3][3] = 1
            # k=5
            gamma[5][0][0] = 1
            # k=6
            gamma[6][2][0] = 1
            gamma[6][3][3] = -1
            
            # Now compute W
            for RC in range(4):
                for pq in range(4):
                    b_coeff = beta[t][pq]
                    g_coeff = gamma[k][RC][pq]
                    total_coeff = b_coeff * g_coeff
                    if total_coeff == 0: continue
                    # C index
                    # RC is block index (0,1) for row, (0,1) for col?
                    # My RC index: 00, 01, 10, 11 -> 0,1,2,3
                    # R = RC // 2, C = RC % 2
                    R = RC // 2
                    C = RC % 2
                    p = pq // 2
                    q = pq % 2
                    c_row = 2*R + p
                    c_col = 2*C + q
                    w_idx = c_col * n + c_row
                    w_vec[w_idx] += total_coeff

            # Add to U, V, W
            for i in range(n * m):
                if u_vec[i] != 0:
                    U[i][col] = u_vec[i]
            for i in range(m * p):
                if v_vec[i] != 0:
                    V[i][col] = v_vec[i]
            for i in range(p * n):
                if w_vec[i] != 0:
                    W[i][col] = w_vec[i]
                    
            col += 1
            
    # --- Part 2: 4x4x1 Schoolbook ---
    # C[:, 4] = A * B[:, 4]
    # C_i4 = sum_j A_ij B_j4
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
