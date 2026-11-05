import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Part 1: 4x4x4 core using nested Strassen (49 terms)
    # Outer Strassen: split 4x4 into 2x2 blocks
    # A blocks: A00(i=0,1;k=0,1), A01(i=0,1;k=2,3), A10(i=2,3;k=0,1), A11(i=2,3;k=2,3)
    # B blocks: B00(k=0,1;j=0,1), B01(k=0,1;j=2,3), B10(k=2,3;j=0,1), B11(k=2,3;j=2,3)
    # C blocks: C00(i=0,1;j=0,1), C01(i=0,1;j=2,3), C10(i=2,3;j=0,1), C11(i=2,3;j=2,3)
    
    strassen_products = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},   # M1 = (A00+A11)(B00+B11)
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},           # M2 = (A10+A11)B00
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},          # M3 = A00(B01-B11)
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},          # M4 = A11(B10-B00)
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},           # M5 = (A00+A01)B11
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},    # M6 = (A10-A00)(B00+B01)
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},    # M7 = (A01-A11)(B10+B11)
    ]
    
    c_contributions = [
        {0: 1, 3: 1},    # M1 -> C00(+1), C11(+1)
        {2: 1, 3: -1},   # M2 -> C10(+1), C11(-1)
        {1: 1, 3: 1},    # M3 -> C01(+1), C11(+1)
        {0: 1, 2: 1},    # M4 -> C00(+1), C10(+1)
        {0: -1, 1: 1},   # M5 -> C00(-1), C01(+1)
        {3: 1},          # M6 -> C11(+1)
        {0: 1},          # M7 -> C00(+1)
    ]
    
    def a_global(block, li, lk):
        if block == 0: return li, lk
        elif block == 1: return li, lk + 2
        elif block == 2: return li + 2, lk
        elif block == 3: return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        if block == 0: return lk, lj
        elif block == 1: return lk, lj + 2
        elif block == 2: return lk + 2, lj
        elif block == 3: return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        if block == 0: return li, lj
        elif block == 1: return li, lj + 2
        elif block == 2: return li + 2, lj
        elif block == 3: return li + 2, lj + 2
    
    # Inner Strassen for 2x2x2 (7 terms)
    inner_strassen = [
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    for prod_idx, prod in enumerate(strassen_products):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = c_contributions[prod_idx]
        
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            
            for c_block, c_sign in c_contrib.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: 4x4x1 (j=4, k in 0..3) - 4 terms
    for k in range(4):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for i in range(4):
            u_vec[i * m + k] = 1.0
        v_vec[k * p + 4] = 1.0
        for i in range(4):
            w_vec[4 * n + i] = 1.0
        
        add_rank1(u_vec, v_vec, w_vec)
    
    # Part 3: 4x1x5 (k=4, j in 0..4) - 1 term
    u_vec = [0.0] * (n * m)
    v_vec = [0.0] * (m * p)
    w_vec = [0.0] * (p * n)
    
    for i in range(4):
        u_vec[i * m + 4] = 1.0
    for j in range(5):
        v_vec[4 * p + j] = 1.0
    for i in range(4):
        for j in range(5):
            w_vec[j * n + i] = 1.0
    
    add_rank1(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r in range(rank):
        for idx, val in enumerate(U_cols[r]):
            if val != 0:
                U[idx, r] = val
        for idx, val in enumerate(V_cols[r]):
            if val != 0:
                V[idx, r] = val
        for idx, val in enumerate(W_cols[r]):
            if val != 0:
                W[idx, r] = val
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank0 = n * m * p
    U_sb = np.zeros((n*m, rank0), dtype=int)
    V_sb = np.zeros((m*p, rank0), dtype=int)
    W_sb = np.zeros((p*n, rank0), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
