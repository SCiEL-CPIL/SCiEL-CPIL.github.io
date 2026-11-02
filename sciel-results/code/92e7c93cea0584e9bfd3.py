import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 66
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0

    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx, col] = val
        V[v_idx, col] = val
        W[w_idx, col] = val
        col += 1

    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx, col] = 1.0
        V[v_idx, col] = 1.0
        W[w_idx, col] = 1.0
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # m1 = (a+d)(e+h)
        U[a_idx, col] = 1; U[d_idx, col] = 1
        V[e_idx, col] = 1; V[h_idx, col] = 1
        W[p_idx, col] = 1; W[s_idx, col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[c_idx, col] = 1; U[d_idx, col] = 1
        V[e_idx, col] = 1
        W[r_idx, col] = 1; W[s_idx, col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[a_idx, col] = 1
        V[f_idx, col] = 1; V[h_idx, col] = -1
        W[q_idx, col] = 1; W[s_idx, col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[d_idx, col] = 1
        V[g_idx, col] = 1; V[e_idx, col] = -1
        W[p_idx, col] = 1; W[r_idx, col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[a_idx, col] = 1; U[b_idx, col] = 1
        V[h_idx, col] = 1
        W[p_idx, col] = -1; W[q_idx, col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[c_idx, col] = 1; U[a_idx, col] = -1
        V[e_idx, col] = 1; V[f_idx, col] = 1
        W[s_idx, col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[b_idx, col] = 1; U[d_idx, col] = -1
        V[g_idx, col] = 1; V[h_idx, col] = 1
        W[p_idx, col] = 1
        col += 1

    # Part 1: Row 0 of A uses schoolbook (4*6 = 24 terms)
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Part 2: Rows 1-2 use Strassen on 2x2x2 sub-blocks
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)
            
    return {"U": U, "V": V, "W": W}
