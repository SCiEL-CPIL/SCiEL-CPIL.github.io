import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, val in u_pairs:
            u_col[idx] = val
        for idx, val in v_pairs:
            v_col[idx] = val
        for idx, val in w_pairs:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen_block(j1, j2, k1, k2):
        # Strassen for 2x2x2: rank 7
        # m1 = (A00+A11)(B00+B11)
        add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
        # m2 = (A10+A11)B00
        add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1)], 
                     [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
        # m3 = A00(B01-B11)
        add_rank_step([(u_idx(0,j1), 1)], 
                     [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                     [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
        # m4 = A11(B10-B00)
        add_rank_step([(u_idx(1,j2), 1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                     [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
        # m5 = (A00+A01)B11
        add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                     [(v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
        # m6 = (A10-A00)(B00+B01)
        add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                     [(w_idx(k2,1), 1)])
        # m7 = (A01-A11)(B10+B11)
        add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1)])

    def add_schoolbook_2x1x2(j, k1, k2):
        # 2 rows, 1 middle, 2 cols: rank 4
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,0), 1)])
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,0), 1)])
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,1), 1)])
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,1), 1)])

    # Partition: j = (0,1), (2,3), (4); k = (0,1), (2,3), (4,5)
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4, 5)]
    
    for j_range in j_groups:
        for k_range in k_groups:
            if len(j_range) == 2 and len(k_range) == 2:
                add_strassen_block(j_range[0], j_range[1], k_range[0], k_range[1])
            elif len(j_range) == 1 and len(k_range) == 2:
                add_schoolbook_2x1x2(j_range[0], k_range[0], k_range[1])
            elif len(j_range) == 2 and len(k_range) == 1:
                # 2x2x1: rank 4
                for i in range(n):
                    for j in j_range:
                        add_rank_step([(u_idx(i,j), 1)], [(v_idx(j,k_range[0]), 1)], [(w_idx(k_range[0],i), 1)])
            else:
                # 2x1x1: rank 2
                for i in range(n):
                    add_rank_step([(u_idx(i,j_range[0]), 1)], [(v_idx(j_range[0],k_range[0]), 1)], [(w_idx(k_range[0],i), 1)])

    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]

    return {"U": U, "V": V, "W": W}
