import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    U = [[0] * 100 for _ in range(n * m)] # Pre-allocate large enough
    V = [[0] * 100 for _ in range(m * p)]
    W = [[0] * 100 for _ in range(p * n)]
    
    rank_counter = [0]
    
    def add_term_direct(u_entries, v_entries, w_entries, i_off, j_off, k_off):
        r = rank_counter[0]
        rank_counter[0] += 1
        for (li, lj, c) in u_entries:
            gi = i_off + li
            gj = j_off + lj
            U[gi * m + gj][r] = c
        for (lj, lk, c) in v_entries:
            gj = j_off + lj
            gk = k_off + lk
            V[gj * p + gk][r] = c
        for (lk, li, c) in w_entries:
            gk = k_off + lk
            gi = i_off + li
            W[gk * n + gi][r] = c

    def decompose_direct(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            add_term_direct([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], i_off, j_off, k_off)
            add_term_direct([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)], i_off, j_off, k_off)
            add_term_direct([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], i_off, j_off, k_off)
            add_term_direct([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], i_off, j_off, k_off)
            add_term_direct([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)], i_off, j_off, k_off)
            add_term_direct([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], i_off, j_off, k_off)
            add_term_direct([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)], i_off, j_off, k_off)
        elif m_sub > 2 and m_sub % 2 == 0:
            m_half = m_sub // 2
            decompose_direct(n_sub, m_half, p_sub, i_off, j_off, k_off)
            decompose_direct(n_sub, m_half, p_sub, i_off, j_off + m_half, k_off)
        elif p_sub > 2 and p_sub % 2 == 0:
            p_half = p_sub // 2
            decompose_direct(n_sub, m_sub, p_half, i_off, j_off, k_off)
            decompose_direct(n_sub, m_sub, p_half, i_off, j_off, k_off + p_half)
        elif n_sub > 2:
            n1 = 2
            n2 = n_sub - 2
            decompose_direct(n1, m_sub, p_sub, i_off, j_off, k_off)
            decompose_direct(n2, m_sub, p_sub, i_off + n1, j_off, k_off)
        else:
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        add_term_direct([(i, j, 1)], [(j, k, 1)], [(k, i, 1)], i_off, j_off, k_off)

    decompose_direct(n, m, p, 0, 0, 0)
    
    rank = rank_counter[0]
    
    # Slice to actual rank
    U_final = [row[:rank] for row in U]
    V_final = [row[:rank] for row in V]
    W_final = [row[:rank] for row in W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
