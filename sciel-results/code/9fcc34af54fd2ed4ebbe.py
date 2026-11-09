"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen on 2x2x2 blocks (rank 117)."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    Uses 2+2+1 block partition with Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    # Strassen 2x2x2 terms: 7 terms
    # Each term is (u_coeffs, v_coeffs, w_coeffs) where keys are (row, col)
    STRASSEN_2x2_TERMS = [
        # m1 = (a00+a11)(b00+b11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2 = (a10+a11)b00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3 = a00(b01-b11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m4 = a11(b10-b00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5 = (a00+a01)b11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # m6 = (a10-a00)(b00+b01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7 = (a01-a11)(b10+b11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_2x2x2(I_map, J_map, K_map):
        for u_coeffs, v_coeffs, w_coeffs in STRASSEN_2x2_TERMS:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), val in u_coeffs.items():
                gi = I_map[li]
                gj = J_map[lj]
                u_entries.append((gi * m + gj, val))
            for (lj, lk), val in v_coeffs.items():
                gj = J_map[lj]
                gk = K_map[lk]
                v_entries.append((gj * p + gk, val))
            for (li, lk), val in w_coeffs.items():
                gi = I_map[li]
                gk = K_map[lk]
                w_entries.append((gk * n + gi, val))
            add_column(u_entries, v_entries, w_entries)

    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )

    def add_block(i_list, j_list, k_list):
        ni, nj, nk = len(i_list), len(j_list), len(k_list)
        
        if ni == 2 and nj == 2 and nk == 2:
            add_strassen_2x2x2(
                {0: i_list[0], 1: i_list[1]},
                {0: j_list[0], 1: j_list[1]},
                {0: k_list[0], 1: k_list[1]}
            )
        elif ni == 1 and nj == 1 and nk == 1:
            schoolbook_block(i_list, j_list, k_list)
        elif ni == 2 and nj == 1 and nk == 1:
            # 2x1x1: rank 2
            a0 = i_list[0] * m + j_list[0]
            a1 = i_list[1] * m + j_list[0]
            b0 = j_list[0] * p + k_list[0]
            c0 = k_list[0] * n + i_list[0]
            c1 = k_list[0] * n + i_list[1]
            add_column([(a0, 1.0)], [(b0, 1.0)], [(c0, 1.0)])
            add_column([(a1, 1.0)], [(b0, 1.0)], [(c1, 1.0)])
        elif ni == 1 and nj == 2 and nk == 2:
            # 1x2x2: rank 4
            a0 = i_list[0] * m + j_list[0]
            a1 = i_list[0] * m + j_list[1]
            b00 = j_list[0] * p + k_list[0]
            b01 = j_list[0] * p + k_list[1]
            b10 = j_list[1] * p + k_list[0]
            b11 = j_list[1] * p + k_list[1]
            c0 = k_list[0] * n + i_list[0]
            c1 = k_list[1] * n + i_list[0]
            add_column([(a0, 1.0)], [(b00, 1.0)], [(c0, 1.0)])
            add_column([(a1, 1.0)], [(b10, 1.0)], [(c0, 1.0)])
            add_column([(a0, 1.0)], [(b01, 1.0)], [(c1, 1.0)])
            add_column([(a1, 1.0)], [(b11, 1.0)], [(c1, 1.0)])
        elif ni == 2 and nj == 2 and nk == 1:
            # 2x2x1: rank 4
            a00 = i_list[0] * m + j_list[0]
            a01 = i_list[0] * m + j_list[1]
            a10 = i_list[1] * m + j_list[0]
            a11 = i_list[1] * m + j_list[1]
            b0 = j_list[0] * p + k_list[0]
            b1 = j_list[1] * p + k_list[0]
            c0 = k_list[0] * n + i_list[0]
            c1 = k_list[0] * n + i_list[1]
            add_column([(a00, 1.0)], [(b0, 1.0)], [(c0, 1.0)])
            add_column([(a01, 1.0)], [(b1, 1.0)], [(c0, 1.0)])
            add_column([(a10, 1.0)], [(b0, 1.0)], [(c1, 1.0)])
            add_column([(a11, 1.0)], [(b1, 1.0)], [(c1, 1.0)])
        elif ni == 2 and nj == 1 and nk == 2:
            # 2x1x2: rank 4
            a0 = i_list[0] * m + j_list[0]
            a1 = i_list[1] * m + j_list[0]
            b00 = j_list[0] * p + k_list[0]
            b01 = j_list[0] * p + k_list[1]
            c00 = k_list[0] * n + i_list[0]
            c01 = k_list[1] * n + i_list[0]
            c10 = k_list[0] * n + i_list[1]
            c11 = k_list[1] * n + i_list[1]
            add_column([(a0, 1.0)], [(b00, 1.0)], [(c00, 1.0)])
            add_column([(a0, 1.0)], [(b01, 1.0)], [(c01, 1.0)])
            add_column([(a1, 1.0)], [(b00, 1.0)], [(c10, 1.0)])
            add_column([(a1, 1.0)], [(b01, 1.0)], [(c11, 1.0)])
        elif ni == 1 and nj == 2 and nk == 1:
            # 1x2x1: rank 2
            a0 = i_list[0] * m + j_list[0]
            a1 = i_list[0] * m + j_list[1]
            b0 = j_list[0] * p + k_list[0]
            b1 = j_list[1] * p + k_list[0]
            c0 = k_list[0] * n + i_list[0]
            add_column([(a0, 1.0)], [(b0, 1.0)], [(c0, 1.0)])
            add_column([(a1, 1.0)], [(b1, 1.0)], [(c0, 1.0)])
        elif ni == 1 and nj == 1 and nk == 2:
            # 1x1x2: rank 2
            a0 = i_list[0] * m + j_list[0]
            b00 = j_list[0] * p + k_list[0]
            b01 = j_list[0] * p + k_list[1]
            c0 = k_list[0] * n + i_list[0]
            c1 = k_list[1] * n + i_list[0]
            add_column([(a0, 1.0)], [(b00, 1.0)], [(c0, 1.0)])
            add_column([(a0, 1.0)], [(b01, 1.0)], [(c1, 1.0)])
        else:
            schoolbook_block(i_list, j_list, k_list)

    # Partition 2+2+1
    parts = [[0, 1], [2, 3], [4]]
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                i_list = parts[bi]
                j_list = parts[bj]
                k_list = parts[bk]
                add_block(i_list, j_list, k_list)

    rank = len(columns)
    
    # Use dict-of-dicts for O(1) sparse insertion, then convert to dense lists
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            if idx not in U_dict:
                U_dict[idx] = {}
            U_dict[idx][c] = val
        for idx, val in v_d.items():
            if idx not in V_dict:
                V_dict[idx] = {}
            V_dict[idx][c] = val
        for idx, val in w_d.items():
            if idx not in W_dict:
                W_dict[idx] = {}
            W_dict[idx][c] = val
    
    U = []
    for i in range(n * m):
        row = [0.0] * rank
        if i in U_dict:
            for c, val in U_dict[i].items():
                row[c] = val
        U.append(row)
    
    V = []
    for i in range(m * p):
        row = [0.0] * rank
        if i in V_dict:
            for c, val in V_dict[i].items():
                row[c] = val
        V.append(row)
    
    W = []
    for i in range(p * n):
        row = [0.0] * rank
        if i in W_dict:
            for c, val in W_dict[i].items():
                row[c] = val
        W.append(row)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
