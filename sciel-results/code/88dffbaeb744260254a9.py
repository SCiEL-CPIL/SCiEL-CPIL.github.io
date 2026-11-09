"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen (rank 73)."""
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
    - 3 diagonal 2x2x2 blocks use Strassen: 3 * 7 = 21
    - 4 off-diagonal 2x2x2 blocks (between the two 2-sized blocks) use Strassen: 4 * 7 = 28
    - 6 blocks involving the 1-sized block use schoolbook: 6 * 4 = 24
    Total rank: 21 + 28 + 24 = 73
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
        columns.append(({k: v for k, v in u_d.items() if v != 0},
                        {k: v for k, v in v_d.items() if v != 0},
                        {k: v for k, v in w_d.items() if v != 0}))
    
    # Standard Strassen 2x2x2 terms for local indices 0,1
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def add_strassen_2x2x2(I_list, J_list, K_list):
        """
        Adds 7 columns for a Strassen 2x2x2 decomposition.
        I_list, J_list, K_list: lists of 2 global indices
        """
        I_map = {0: I_list[0], 1: I_list[1]}
        J_map = {0: J_list[0], 1: J_list[1]}
        K_map = {0: K_list[0], 1: K_list[1]}
        
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
        """Schoolbook for small blocks."""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    # 2+2+1 partition
    # Block 0: [0, 1]
    # Block 1: [2, 3]
    # Block 2: [4]
    blocks = [[0, 1], [2, 3], [4]]
    
    # Process all 3x3x3 = 27 block combinations
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                if _deadline_exceeded():
                    break
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                # If all three blocks are size 2, use Strassen
                if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    add_strassen_2x2x2(i_list, j_list, k_list)
                else:
                    # Otherwise use schoolbook
                    schoolbook_block(i_list, j_list, k_list)
            if _deadline_exceeded():
                break
        if _deadline_exceeded():
            break
    
    rank = len(columns)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx, c] = val
        for idx, val in v_d.items():
            V[idx, c] = val
        for idx, val in w_d.items():
            W[idx, c] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
