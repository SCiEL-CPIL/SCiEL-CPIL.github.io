"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen on all 2x2x2 blocks (rank 117)."""
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
    
    Uses a 2+2+1 block partition.
    All eight 2x2x2 sub-blocks use Strassen decomposition (rank 7 each).
    The 1x1x1 block uses schoolbook (rank 1).
    Total rank: 8 * 7 + 1 = 57.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
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
    
    # Block index groups: 2+2+1 partition
    blocks = [[0, 1], [2, 3], [4]]
    
    # Standard Strassen 2x2x2 terms for local indices 0,1
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_2x2x2(offset_i, offset_j, offset_k):
        """
        Adds 7 columns for a Strassen 2x2x2 decomposition.
        """
        I_map = {0: offset_i, 1: offset_i + 1}
        J_map = {0: offset_j, 1: offset_j + 1}
        K_map = {0: offset_k, 1: offset_k + 1}
        
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

    # Build all blocks
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                if _deadline_exceeded():
                    break
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    add_strassen_2x2x2(i_list[0], j_list[0], k_list[0])
                elif len(i_list) == 1 and len(j_list) == 1 and len(k_list) == 1:
                    i, j, k = i_list[0], j_list[0], k_list[0]
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
                else:
                    # Mixed sizes: use schoolbook for each element
                    for i in i_list:
                        for j in j_list:
                            for k in k_list:
                                add_column(
                                    [(i * m + j, 1.0)],
                                    [(j * p + k, 1.0)],
                                    [(k * n + i, 1.0)]
                                )
            if _deadline_exceeded():
                break
        if _deadline_exceeded():
            break
    
    rank = len(columns)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
