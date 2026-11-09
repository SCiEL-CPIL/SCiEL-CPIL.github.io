import numpy as np
import time

DIMENSIONS = (4, 4, 7)

# Strassen coefficients for 2x2 multiplication
# Each entry is (U_coeffs, V_coeffs, W_coeffs)
# Coeffs are lists of (row, col, value)
STRASSEN_2X2 = [
    # M1 = (a+d)(e+h)
    ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
    # M2 = (c+d)e
    ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
    # M3 = a(f-h)
    ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
    # M4 = d(g-e)
    ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
    # M5 = (a+b)h
    ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
    # M6 = (c-a)(e+f)
    ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
    # M7 = (b-d)(g+h)
    ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    RANK_444 = 49
    RANK_443 = 44
    TOTAL_RANK = RANK_444 + RANK_443
    
    U = np.zeros((nm, TOTAL_RANK), dtype=np.float64)
    V = np.zeros((mp, TOTAL_RANK), dtype=np.float64)
    W = np.zeros((pn, TOTAL_RANK), dtype=np.float64)
    
    # --- Part 1: 4x4x4 Decomposition (Rank 49) ---
    for s_outer in range(7):
        u_outer, v_outer, w_outer = STRASSEN_2X2[s_outer]
        for s_inner in range(7):
            u_inner, v_inner, w_inner = STRASSEN_2X2[s_inner]
            r_idx = s_outer * 7 + s_inner
            
            u_col = np.zeros(nm)
            v_col = np.zeros(mp)
            w_col = np.zeros(pn)
            
            for (bx, by, coeff_o) in u_outer:
                for (ir, ic, coeff_i) in u_inner:
                    i = 2 * bx + ir
                    j = 2 * by + ic
                    a_idx = i * m + j
                    u_col[a_idx] += coeff_o * coeff_i
            
            for (bx, by, coeff_o) in v_outer:
                for (ir, ic, coeff_i) in v_inner:
                    i = 2 * bx + ir
                    j = 2 * by + ic
                    b_idx = i * p + j
                    v_col[b_idx] += coeff_o * coeff_i
            
            for (bx, by, coeff_o) in w_outer:
                for (ir, ic, coeff_i) in w_inner:
                    i = 2 * bx + ir
                    j = 2 * by + ic
                    c_idx = j * n + i
                    w_col[c_idx] += coeff_o * coeff_i
            
            U[:, r_idx] = u_col
            V[:, r_idx] = v_col
            W[:, r_idx] = w_col
    
    # --- Part 2: 4x4x3 Decomposition (Rank 44) ---
    offset = RANK_444
    col_idx = 0
    
    subproblems = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (2, 0, 0, 2),
        (2, 2, 2, 2)
    ]
    
    for (ar, ac, br, cr) in subproblems:
        # Strassen for first 2 columns (local cols 0,1 -> global 4,5)
        for s_idx in range(7):
            u_coeff, v_coeff, w_coeff = STRASSEN_2X2[s_idx]
            
            u_col = np.zeros(nm)
            v_col = np.zeros(mp)
            w_col = np.zeros(pn)
            
            for (ir, ic, c_val) in u_coeff:
                i = ar + ir
                j = ac + ic
                a_idx = i * m + j
                u_col[a_idx] += c_val
            
            for (ir, ic, c_val) in v_coeff:
                i = br + ir
                j = 4 + ic
                b_idx = i * p + j
                v_col[b_idx] += c_val
            
            for (ir, ic, c_val) in w_coeff:
                i = cr + ir
                k = 4 + ic
                c_idx = k * n + i
                w_col[c_idx] += c_val
            
            U[:, offset + col_idx] = u_col
            V[:, offset + col_idx] = v_col
            W[:, offset + col_idx] = w_col
            col_idx += 1
        
        # Schoolbook for 3rd column (local col 2 -> global 6)
        for ir in range(2):
            for jc in range(2):
                u_col = np.zeros(nm)
                v_col = np.zeros(mp)
                w_col = np.zeros(pn)
                
                i_a = ar + ir
                j_a = ac + jc
                a_idx = i_a * m + j_a
                u_col[a_idx] = 1.0
                
                i_b = br + jc
                j_b = 6
                b_idx = i_b * p + j_b
                v_col[b_idx] = 1.0
                
                i_c = cr + ir
                k_c = 6
                c_idx = k_c * n + i_c
                w_col[c_idx] = 1.0
                
                U[:, offset + col_idx] = u_col
                V[:, offset + col_idx] = v_col
                W[:, offset + col_idx] = w_col
                col_idx += 1
    
    # Light validation
    if time.time() - start_time < 100:
        np.random.seed(42)
        for _ in range(100):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            a = i * m + j
            b = j * p + k
            c = k * n + i
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-9:
                raise ValueError(f"Validation failed target ({a},{b},{c}): {val}")
        
        for _ in range(100):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            i_a, j_a = divmod(a, m)
            j_b, k_b = divmod(b, p)
            k_c, i_c = divmod(c, n)
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    raise ValueError(f"Validation failed non-target ({a},{b},{c}): {val}")
    
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
