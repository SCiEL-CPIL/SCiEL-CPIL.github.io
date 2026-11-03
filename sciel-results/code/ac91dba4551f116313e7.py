import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # 8 blocks of Strassen
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition
    # Each term is a tuple (u_list, v_list, w_list)
    # u_list: list of (local_u, coeff)
    # local_u: 0=A00, 1=A01, 2=A10, 3=A11
    # v_list: list of (local_v, coeff)
    # local_v: 0=B00, 1=B01, 2=B10, 3=B11
    # w_list: list of (local_w, coeff)
    # local_w: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = [
        # M1 = (A00+A11)(B00+B11)
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)] # C00 +1, C11 +1
        },
        # M2 = (A10+A11)B00
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(2, 1), (3, -1)] # C10 +1, C11 -1
        },
        # M3 = A00(B01-B11)
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)] # C01 +1, C11 +1
        },
        # M4 = A11(B10-B00)
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)] # C00 +1, C10 +1
        },
        # M5 = (A00+A01)B11
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(0, -1), (1, 1)] # C00 -1, C01 +1
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)] # C11 +1
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)] # C00 +1
        }
    ]
    
    # Define the 8 blocks
    # (a_col, b_row, b_col, c_col)
    blocks = [
        (0, 0, 0, 0), # A0 * B0_top -> C0
        (0, 0, 2, 2), # A0 * B1_top -> C1
        (0, 0, 4, 4), # A0 * B2_top -> C2
        (0, 0, 6, 6), # A0 * B3_top -> C3
        (2, 2, 0, 0), # A1 * B0_bot -> C0
        (2, 2, 2, 2), # A1 * B1_bot -> C1
        (2, 2, 4, 4), # A1 * B2_bot -> C2
        (2, 2, 6, 6), # A1 * B3_bot -> C3
    ]
    
    col = 0
    for a_col, b_row, b_col, c_col in blocks:
        for term in terms:
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            # Fill U
            for lu, coeff in term['u']:
                i = lu // 2
                jl = lu % 2
                # Global U index: i * m + (a_col + jl)
                gu = i * m + (a_col + jl)
                U[gu][col] += coeff
            
            # Fill V
            for lv, coeff in term['v']:
                jr = lv // 2
                kl = lv % 2
                # Global V index: (b_row + jr) * p + (b_col + kl)
                gv = (b_row + jr) * p + (b_col + kl)
                V[gv][col] += coeff
            
            # Fill W
            for lw, coeff in term['w']:
                i = lw // 2
                kl = lw % 2
                # Global W index: (c_col + kl) * n + i
                gw = (c_col + kl) * n + i
                W[gw][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
    
    # Strassen terms definition
    # Each term is a tuple (u_list, v_list, w_list)
    # u_list: list of (local_u, coeff)
    # local_u: 0=A00, 1=A01, 2=A10, 3=A11
    # v_list: list of (local_v, coeff)
    # local_v: 0=B00, 1=B01, 2=B10, 3=B11
    # w_list: list of (local_w, coeff)
    # local_w: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = [
        # M1 = (A00+A11)(B00+B11)
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)] # C00 +1, C11 +1
        },
        # M2 = (A10+A11)B00
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(2, 1), (3, -1)] # C10 +1, C11 -1
        },
        # M3 = A00(B01-B11)
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)] # C01 +1, C11 +1
        },
        # M4 = A11(B10-B00)
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)] # C00 +1, C10 +1
        },
        # M5 = (A00+A01)B11
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(0, -1), (1, 1)] # C00 -1, C01 +1
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)] # C11 +1
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)] # C00 +1
        }
    ]
    
    # Define the 8 blocks
    # (a_col, b_row, b_col, c_col)
    blocks = [
        (0, 0, 0, 0), # A0 * B0_top -> C0
        (0, 0, 2, 2), # A0 * B1_top -> C1
        (0, 0, 4, 4), # A0 * B2_top -> C2
        (0, 0, 6, 6), # A0 * B3_top -> C3
        (2, 2, 0, 0), # A1 * B0_bot -> C0
        (2, 2, 2, 2), # A1 * B1_bot -> C1
        (2, 2, 4, 4), # A1 * B2_bot -> C2
        (2, 2, 6, 6), # A1 * B3_bot -> C3
    ]
    
    col = 0
    for a_col, b_row, b_col, c_col in blocks:
        for term in terms:
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            # Fill U
            for lu, coeff in term['u']:
                i = lu // 2
                jl = lu % 2
                # Global U index: i * m + (a_col + jl)
                gu = i * m + (a_col + jl)
                U[gu][col] += coeff
            
            # Fill V
            for lv, coeff in term['v']:
                jr = lv // 2
                kl = lv % 2
                # Global V index: (b_row + jr) * p + (b_col + kl)
                gv = (b_row + jr) * p + (b_col + kl)
                V[gv][col] += coeff
            
            # Fill W
            for lw, coeff in term['w']:
                i = lw // 2
                kl = lw % 2
                # Global W index: (c_col + kl) * n + i
                gw = (c_col + kl) * n + i
                W[gw][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
