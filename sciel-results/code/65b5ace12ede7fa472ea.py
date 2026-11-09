import numpy as np
import time

DIMENSIONS = (4, 4, 7)

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    # Strassen 2x2 definitions
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    rank_444 = 49
    rank_443 = 44
    total_rank = rank_444 + rank_443
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # Part 1: 4x4 x 4x4 using Strassen-of-Strassen (rank 49)
    for s_idx, (X_def, Y_def, Z_def) in enumerate(strassen_2x2):
        for w_idx, (u_def, v_def, out_def) in enumerate(strassen_2x2):
            r = s_idx * 7 + w_idx
            
            U_col = np.zeros(nm)
            for (bx, by, cx) in X_def:
                for (i_prime, j_prime, cu) in u_def:
                    i = 2 * bx + i_prime
                    j = 2 * by + j_prime
                    U_col[i * m + j] += cx * cu
            U[:, r] = U_col
            
            V_col = np.zeros(mp)
            for (bx, by, cy) in Y_def:
                for (i_prime, j_prime, cv) in v_def:
                    row_b = 2 * bx + i_prime
                    col_b = 2 * by + j_prime
                    V_col[row_b * p + col_b] += cy * cv
            V[:, r] = V_col
            
            W_col = np.zeros(pn)
            for (dx, dy, cd) in Z_def:
                for (i_prime, j_prime, co) in out_def:
                    row_c = 2 * dx + i_prime
                    col_c = 2 * dy + j_prime
                    W_col[col_c * n + row_c] += cd * co
            W[:, r] = W_col
    
    # Part 2: 4x4 x 4x3 (cols 4-6 of B), rank 44
    offset = rank_444
    col = 0
    
    subproblems = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (2, 0, 0, 2),
        (2, 2, 2, 2),
    ]
    
    for (ar, ac, br, cr) in subproblems:
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_2x2:
            U_col = np.zeros(nm)
            V_col = np.zeros(mp)
            W_col = np.zeros(pn)
            
            for (lr, lc, coeff) in u_coeffs:
                U_col[(ar + lr) * m + (ac + lc)] += coeff
            
            for (lr, lc, coeff) in v_coeffs:
                V_col[(br + lr) * p + (4 + lc)] += coeff
            
            for (lr, lc, coeff) in w_coeffs:
                W_col[(4 + lc) * n + (cr + lr)] += coeff
            
            U[:, offset + col] = U_col
            V[:, offset + col] = V_col
            W[:, offset + col] = W_col
            col += 1
        
        for lr in range(2):
            for j in range(2):
                a_idx = (ar + lr) * m + (ac + j)
                b_idx = (br + j) * p + 6
                c_idx = 6 * n + (cr + lr)
                
                U[a_idx, offset + col] = 1.0
                V[b_idx, offset + col] = 1.0
                W[c_idx, offset + col] = 1.0
                col += 1
    
    # Quick validation of target entries
    if time.time() - start_time < DEADLINE - 5:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        raise ValueError(f"Validation failed for target ({a},{b},{c}): {val}")
    
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
