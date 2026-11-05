import numpy as np
import time
import sys

_DEADLINE = time.time() + 105  # Safety margin before 120s limit

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank: 49 (Strassen 4x4x4) + 44 (4 * 11 for 4x4x3 via 2x2x3 blocks)
    total_rank = 93
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Strassen 2x2x2 terms: (x_sub, y_sub, cm_coeffs)
    # x_sub, y_sub: dict {(li,lj): coeff}
    # cm_coeffs: dict {(li,lj): coeff} for C block
    strassen_2x2x2 = [
        # M1 = (a00+a11)(b00+b11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # M2 = (a10+a11) b00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        # M3 = a00 (b01+b11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        # M4 = a11 (b10-b00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        # M5 = (a00+a01) b11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        # M6 = (a10-a00)(b00+b01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # M7 = (a01-a11)(b10+b11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # High-level Strassen terms for 4x4:
    high_level = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),  # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(1,0): 1, (1,1): -1}),  # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),  # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),  # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (0,1): 1}),  # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(1,1): 1}),  # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): 1}),  # M7
    ]
    
    for h_idx in range(7):
        A_coeffs, B_coeffs, C_coeffs = high_level[h_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, cm_low = strassen_2x2x2[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            # U: A entries
            for (ab_r, ab_c), a_c in A_coeffs.items():
                for (li, lj), x_c in x_sub.items():
                    i = 2 * ab_r + li
                    j = 2 * ab_c + lj
                    u_coeffs[i * m + j] += a_c * x_c
            
            # V: B entries
            for (bb_r, bb_c), b_c in B_coeffs.items():
                for (li, lj), y_c in y_sub.items():
                    i = 2 * bb_r + li
                    j = 2 * bb_c + lj
                    v_coeffs[i * p + j] += b_c * y_c
            
            # W: C entries
            w_coeffs = [0.0] * (p*n)
            for (cb_r, cb_c), c_c in C_coeffs.items():
                for (li, lj), cm_c in cm_low.items():
                    i = 2 * cb_r + li
                    k = 2 * cb_c + lj
                    w_coeffs[k * n + i] += c_c * cm_c
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            for i in range(p*n):
                W[i][r] = w_coeffs[i]
    
    # --- Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44) ---
    # Columns 4,5,6 of B
    # 4 blocks: A00*B_top, A01*B_bot, A10*B_top, A11*B_bot
    
    def add_2x2x3(a_rb, a_cb, b_rb, c_rb):
        nonlocal current_rank
        
        # Strassen for cols 4,5 (2x2x2)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, cm_low = strassen_2x2x2[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), x_c in x_sub.items():
                i = 2 * a_rb + li
                j = 2 * a_cb + lj
                u_coeffs[i * m + j] += x_c
            
            for (li, lj), y_c in y_sub.items():
                i = 2 * b_rb + li
                j = 4 + lj  # cols 4,5
                v_coeffs[i * p + j] += y_c
            
            w_coeffs = [0.0] * (p*n)
            for (li, lj), cm_c in cm_low.items():
                i = 2 * c_rb + li
                k = 4 + lj  # cols 4,5
                w_coeffs[k * n + i] += cm_c
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            for i in range(p*n):
                W[i][r] = w_coeffs[i]
        
        # Schoolbook for col 6 (4 terms)
        schoolbook = [
            ((0,0), (0,0), 0),
            ((0,1), (1,0), 0),
            ((1,0), (0,0), 1),
            ((1,1), (1,0), 1),
        ]
        for a_pos, b_pos, c_row in schoolbook:
            r = current_rank
            current_rank += 1
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            i_a = 2 * a_rb + a_pos[0]
            j_a = 2 * a_cb + a_pos[1]
            u_coeffs[i_a * m + j_a] = 1.0
            
            i_b = 2 * b_rb + b_pos[0]
            j_b = 6
            v_coeffs[i_b * p + j_b] = 1.0
            
            w_coeffs = [0.0] * (p*n)
            i_final = 2 * c_rb + c_row
            k_final = 6
            w_coeffs[k_final * n + i_final] = 1.0
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            for i in range(p*n):
                W[i][r] = w_coeffs[i]
    
    add_2x2x3(0, 0, 0, 0)  # A00 * B_top -> C_top
    add_2x2x3(0, 1, 1, 0)  # A01 * B_bot -> C_top
    add_2x2x3(1, 0, 0, 1)  # A10 * B_top -> C_bot
    add_2x2x3(1, 1, 1, 1)  # A11 * B_bot -> C_bot
    
    # Trim to actual rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if time.time() < _DEADLINE:
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        max_err = np.max(np.abs(recon - target))
        
        if max_err > 1e-10:
            # Fallback: schoolbook
            rank_sb = n * m * p
            U_sb = np.zeros((n*m, rank_sb))
            V_sb = np.zeros((m*p, rank_sb))
            W_sb = np.zeros((p*n, rank_sb))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j, r] = 1.0
                        V_sb[j*p+k, r] = 1.0
                        W_sb[k*n+i, r] = 1.0
                        r += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U, "V": V, "W": W}
