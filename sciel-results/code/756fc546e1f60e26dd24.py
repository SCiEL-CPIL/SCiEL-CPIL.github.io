import numpy as np

def construct():
    n, m, p = 4, 4, 5
    # Rank 49 for 4x4x4
    # Plus 16 for 5th column
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 decomposition (Rank 49)
    # Using Strassen 2x2x2 (Rank 7) applied to blocks
    
    # Strassen factors for 2x2x2
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    U_S = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, -1, -1]
    ])
    V_S = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, -1, 0, 1, 0]
    ])
    
    # Block coefficients for 4x4x4
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A blocks: A11(0,0), A12(0,1), A21(1,0), A22(1,1)
    # B blocks: B11(0,0), B12(0,1), B21(1,0), B22(1,1)
    
    # Coefficients for A blocks for each M_m
    # A_m is 2x2 matrix of coefficients for A11..A22
    # Let's define A_coeff[m] as 2x2x2x2? No, A_m is a 2x2 matrix of linear combos.
    # A_m[p,q] = sum alpha A_block[i,j]
    
    A_coeffs = [
        # M1: A11+A22
        [[1, 0], [0, 1]], 
        # M2: A21+A22
        [[0, 0], [1, 1]], 
        # M3: A11
        [[1, 0], [0, 0]], 
        # M4: A22
        [[0, 0], [0, 1]], 
        # M5: A11+A12
        [[1, 1], [0, 0]], 
        # M6: A21-A11
        [[-1, 0], [1, 0]], 
        # M7: A12-A22
        [[0, 1], [0, -1]] 
    ]
    
    B_coeffs = [
        # M1: B11+B22
        [[1, 0], [0, 1]], 
        # M2: B11
        [[1, 0], [0, 0]], 
        # M3: B12-B22
        [[0, 1], [0, -1]], 
        # M4: B21-B11
        [[-1, 0], [1, 0]], 
        # M5: B22
        [[0, 0], [0, 1]], 
        # M6: B11+B12
        [[1, 1], [0, 0]], 
        # M7: B21+B22
        [[0, 0], [1, 1]] 
    ]
    
    C_coeffs = [
        # C11 = M1 + M4 - M5 + M7
        [1, 0, 0, 1, -1, 0, 1],
        # C12 = M3 + M5
        [0, 0, 1, 0, 1, 0, 0],
        # C21 = M2 + M4
        [0, 1, 0, 1, 0, 0, 0],
        # C22 = M1 - M2 + M3 + M6
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col = 0
    for m in range(7):
        # For each of the 7 Strassen terms s
        for s in range(7):
            # U part
            # U_S[s, i] corresponds to A_block element i (0..3)
            # A_block element i corresponds to A_block[i//2, i%2]
            # A_block[i//2, i%2] = sum A_coeffs[m][i//2, i%2][j,k] A_block[j,k]
            # A_block[j,k] corresponds to A indices:
            # j=0,k=0 -> A[0:2, 0:2] -> U indices 0..3
            # j=0,k=1 -> A[0:2, 2:4] -> U indices 4..7
            # j=1,k=0 -> A[2:4, 0:2] -> U indices 8..11
            # j=1,k=1 -> A[2:4, 2:4] -> U indices 12..15
            
            # V part similar for B
            # W part similar for C
            
            for i in range(4): # A_block elements
                u_val = U_S[s, i]
                if u_val != 0:
                    # i corresponds to A_block[i//2, i%2]
                    bi = i // 2
                    bj = i % 2
                    # A_coeffs[m][bi, bj] is the coefficient for A_block[bi, bj]
                    # Wait, A_coeffs[m] is 2x2 matrix of coeffs for A11..A22
                    # So A_coeffs[m][bi, bj] is the scalar coeff
                    coeff_a = A_coeffs[m][bi, bj]
                    if coeff_a != 0:
                        # Add to U
                        # A_block[bi, bj] occupies U rows:
                        # base = (bi * 2 + bj) * 4 ?
                        # A is 4x4.
                        # Block (0,0) is A[0:2, 0:2]. Rows 0,1. Cols 0,1.
                        # U index = r * 4 + c.
                        # For block (bi, bj):
                        # r in {2*bi, 2*bi+1}
                        # c in {2*bj, 2*bj+1}
                        for dr in range(2):
                            for dc in range(2):
                                r = 2 * bi + dr
                                c = 2 * bj + dc
                                u_idx = r * 4 + c
                                U[u_idx][col] += u_val * coeff_a
            
            for k in range(4): # B_block elements
                v_val = V_S[s, k]
                if v_val != 0:
                    bk = k // 2
                    bl = k % 2
                    coeff_b = B_coeffs[m][bk, bl]
                    if coeff_b != 0:
                        # B is 4x4 (first 4 cols).
                        # Block (bk, bl) occupies V rows:
                        # r in {2*bk, 2*bk+1}
                        # c in {2*bl, 2*bl+1}
                        # V index = r * 4 + c ?
                        # V shape (16, R) for 4x4.
                        # But V is 4x5.
                        # For the 4x4 part, cols 0..3.
                        for dr in range(2):
                            for dc in range(2):
                                r = 2 * bk + dr
                                c = 2 * bl + dc
                                v_idx = r * 4 + c
                                V[v_idx][col] += v_val * coeff_b
            
            for w_idx_out in range(4): # C_block elements
                # C_block element w_idx_out corresponds to C_block[w_idx_out//2, w_idx_out%2]
                cw = w_idx_out // 2
                cx = w_idx_out % 2
                # C_coeffs[cw, cx] is the coeff for M_m
                coeff_c = C_coeffs[cw, cx]
                if coeff_c != 0:
                    w_val = W_S[s, w_idx_out]
                    if w_val != 0:
                        # C is 4x4.
                        # Block (cw, cx) occupies W rows:
                        # r in {2*cw, 2*cw+1}
                        # c in {2*cx, 2*cx+1}
                        # W index = c * 4 + r ?
                        # W[k*n+i]. k is col, i is row.
                        # For C_block(cw, cx):
                        # col k in {2*cx, 2*cx+1}
                        # row i in {2*cw, 2*cw+1}
                        for dr in range(2):
                            for dc in range(2):
                                k = 2 * cx + dc
                                i = 2 * cw + dr
                                w_idx = k * 4 + i
                                W[w_idx][col] += w_val * coeff_c
            
            col += 1
            
    # Part 2: 5th column (Rank 16)
    # C[:, 4] = A B[:, 4]
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
