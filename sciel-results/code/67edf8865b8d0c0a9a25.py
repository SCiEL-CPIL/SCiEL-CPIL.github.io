import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Ranks:
    # 2x2x7: 3*7 (Strassen) + 4 (Schoolbook) = 25
    # 2x1x7: 7
    # 1x2x7: 7
    # 1x1x7: 7
    total_rank = 46
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, u_vals=None, v_vals=None, w_vals=None):
        nonlocal col
        if u_vals is None: u_vals = [1]*len(u_indices)
        if v_vals is None: v_vals = [1]*len(v_indices)
        if w_vals is None: w_vals = [1]*len(w_indices)
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    def add_strassen_222(gi, gj, gk):
        # gi: [i0, i1], gj: [j0, j1], gk: [k0, k1]
        # M1: (A11+A22)(B11+B22)
        add_term([gi[0]*m + gj[0], gi[1]*m + gj[1]], 
                 [gj[0]*p + gk[0], gj[1]*p + gk[1]], 
                 [gk[0]*n + gi[0], gk[1]*n + gi[1]])
        # M2: (A21+A22)B11
        add_term([gi[1]*m + gj[0], gi[1]*m + gj[1]], 
                 [gj[0]*p + gk[0]], 
                 [gk[0]*n + gi[1]])
        # M3: A11(B21-B22)
        add_term([gi[0]*m + gj[0]], 
                 [gj[1]*p + gk[0], gj[1]*p + gk[1]], 
                 [gk[1]*n + gi[0], gk[1]*n + gi[1]],
                 v_vals=[1, -1])
        # M4: A22(B12-B11)
        add_term([gi[1]*m + gj[1]], 
                 [gj[0]*p + gk[1], gj[0]*p + gk[0]], 
                 [gk[0]*n + gi[1], gk[1]*n + gi[1]],
                 v_vals=[1, -1], w_vals=[1, -1])
        # M5: (A11+A12)B22
        add_term([gi[0]*m + gj[0], gi[0]*m + gj[1]], 
                 [gj[1]*p + gk[1]], 
                 [gk[1]*n + gi[0], gk[0]*n + gi[0]],
                 w_vals=[1, -1])
        # M6: (A21-A11)(B11+B12)
        add_term([gi[1]*m + gj[0], gi[0]*m + gj[0]], 
                 [gj[0]*p + gk[0], gj[0]*p + gk[1]], 
                 [gk[1]*n + gi[1]],
                 u_vals=[1, -1])
        # M7: (A12-A22)(B21+B22)
        add_term([gi[0]*m + gj[1], gi[1]*m + gj[1]], 
                 [gj[1]*p + gk[0], gj[1]*p + gk[1]], 
                 [gk[0]*n + gi[0], gk[1]*n + gi[1]],
                 u_vals=[1, -1])

    def add_schoolbook_221(gi, gj, gk):
        for i in range(2):
            for j in range(2):
                add_term([gi[i]*m + gj[j]], 
                         [gj[j]*p + gk], 
                         [gk*n + gi[i]])

    def add_217(gi_list, gj, gk_list):
        # 2x1x7: Rank 7
        for k in gk_list:
            add_term([gi_list[0]*m + gj, gi_list[1]*m + gj], 
                     [gj*p + k], 
                     [k*n + gi_list[0], k*n + gi_list[1]])

    def add_127(gi, gj_list, gk_list):
        # 1x2x7: Rank 7
        for k in gk_list:
            add_term([gi*m + gj_list[0], gi*m + gj_list[1]], 
                     [gj_list[0]*p + k, gj_list[1]*p + k], 
                     [k*n + gi])

    def add_117(gi, gj, gk_list):
        # 1x1x7: Rank 7
        for k in gk_list:
            add_term([gi*m + gj], 
                     [gj*p + k], 
                     [k*n + gi])

    gi_2 = [0, 1]
    gj_2 = [0, 1]
    gi_1 = [2]
    gj_1 = [2]
    gk_all = list(range(7))
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:7]
    add_strassen_222(gi_2, gj_2, [0, 1])
    add_strassen_222(gi_2, gj_2, [2, 3])
    add_strassen_222(gi_2, gj_2, [4, 5])
    add_schoolbook_221(gi_2, gj_2, 6)
    
    # Part 2: A[0:2, 2] * B[2, 0:7]
    add_217(gi_2, 2, gk_all)
    
    # Part 3: A[2, 0:2] * B[0:2, 0:7]
    add_127(2, gj_2, gk_all)
    
    # Part 4: A[2, 2] * B[2, 0:7]
    add_117(2, 2, gk_all)
    
    return {"U": U, "V": V, "W": W}
