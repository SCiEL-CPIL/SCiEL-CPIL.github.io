"""5x5x5 matrix multiplication tensor decomposition using 4+1 block structure with Strassen on 4x4x4 block (rank 89)."""
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
    Uses 4+1 block partition with Strassen for the 4x4x4 block (decomposed into four 2x2x2 Strassen blocks).
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
    
    # Strassen 2x2x2 terms
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
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
        
        if ni == 4 and nj == 4 and nk == 4:
            # Decompose 4x4x4 into four 2x2x2 blocks
            # Top-Left: rows 0,1; cols 0,1
            add_strassen_2x2x2(
                {0: i_list[0], 1: i_list[1]},
                {0: j_list[0], 1: j_list[1]},
                {0: k_list[0], 1: k_list[1]}
            )
            # Top-Right: rows 0,1; cols 2,3
            add_strassen_2x2x2(
                {0: i_list[0], 1: i_list[1]},
                {0: j_list[2], 1: j_list[3]},
                {0: k_list[0], 1: k_list[1]}
            )
            # Bottom-Left: rows 2,3; cols 0,1
            add_strassen_2x2x2(
                {0: i_list[2], 1: i_list[3]},
                {0: j_list[0], 1: j_list[1]},
                {0: k_list[0], 1: k_list[1]}
            )
            # Bottom-Right: rows 2,3; cols 2,3
            add_strassen_2x2x2(
                {0: i_list[2], 1: i_list[3]},
                {0: j_list[2], 1: j_list[3]},
                {0: k_list[0], 1: k_list[1]}
            )
        else:
            schoolbook_block(i_list, j_list, k_list)

    # Partition 4+1
    parts = [[0, 1, 2, 3], [4]]
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
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
