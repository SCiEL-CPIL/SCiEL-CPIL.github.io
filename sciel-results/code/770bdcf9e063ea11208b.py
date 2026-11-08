import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 44
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Local 3x2x2 decomposition
    # U_loc: 6x11 (3 rows, 2 cols)
    # V_loc: 4x11 (2 rows, 2 cols)
    # W_loc: 6x11 (2 cols, 3 rows) -> indexed by k, i
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Indices for local X (3x2): i in 0..2, j' in 0..1
    # Indices for local Y (2x2): j' in 0..1, k' in 0..1
    # Indices for local Z (3x2): i in 0..2, k' in 0..1
    
    # Strassen part (7 mults)
    # M1 = (x00 + x11)(y00 + y11)
    U_loc[0, 0] += 1  # x00
    U_loc[3, 0] += 1  # x11
    V_loc[0, 0] += 1  # y00
    V_loc[3, 0] += 1  # y11
    W_loc[0, 0] += 1  # z00 (k'=0, i=0)
    W_loc[3, 0] += 1  # z11 (k'=1, i=1)
    
    # M2 = (x10 + x11)y00
    U_loc[2, 1] += 1  # x10
    U_loc[3, 1] += 1  # x11
    V_loc[0, 1] += 1  # y00
    W_loc[2, 1] += 1  # z10 (k'=0, i=1)
    
    # M3 = x00(y10 - y00)
    U_loc[0, 2] += 1  # x00
    V_loc[2, 2] += 1  # y10
    V_loc[0, 2] -= 1  # y00
    W_loc[1, 2] += 1  # z01 (k'=1, i=0)
    
    # M4 = x11(y01 - y11)
    U_loc[3, 3] += 1  # x11
    V_loc[1, 3] += 1  # y01
    V_loc[3, 3] -= 1  # y11
    W_loc[0, 3] += 1  # z00 (k'=0, i=0)
    W_loc[2, 3] += 1  # z10 (k'=0, i=1)
    
    # M5 = (x00 + x01)y11
    U_loc[0, 4] += 1  # x00
    U_loc[1, 4] += 1  # x01
    V_loc[3, 4] += 1  # y11
    W_loc[0, 4] -= 1  # z00 (k'=0, i=0)
    W_loc[1, 4] += 1  # z01 (k'=1, i=0)
    
    # M6 = (x10 - x00)(y00 + y01)
    U_loc[2, 5] += 1  # x10
    U_loc[0, 5] -= 1  # x00
    V_loc[0, 5] += 1  # y00
    V_loc[1, 5] += 1  # y01
    W_loc[3, 5] += 1  # z11 (k'=1, i=1)
    
    # M7 = (x01 - x11)(y10 + y11)
    U_loc[1, 6] += 1  # x01
    U_loc[3, 6] -= 1  # x11
    V_loc[2, 6] += 1  # y10
    V_loc[3, 6] += 1  # y11
    W_loc[0, 6] += 1  # z00 (k'=0, i=0)
    
    # Dot product part (4 mults)
    # M8 = x20 y00
    U_loc[4, 7] += 1  # x20
    V_loc[0, 7] += 1  # y00
    W_loc[4, 7] += 1  # z20 (k'=0, i=2)
    
    # M9 = x21 y10
    U_loc[5, 8] += 1  # x21
    V_loc[2, 8] += 1  # y10
    W_loc[4, 8] += 1  # z20 (k'=0, i=2)
    
    # M10 = x20 y01
    U_loc[4, 9] += 1  # x20
    V_loc[1, 9] += 1  # y01
    W_loc[5, 9] += 1  # z21 (k'=1, i=2)
    
    # M11 = x21 y11
    U_loc[5, 10] += 1  # x21
    V_loc[3, 10] += 1  # y11
    W_loc[5, 10] += 1  # z21 (k'=1, i=2)
    
    # Map local to global for 4 blocks
    # Block 1: j in {0,1}, k in {0,1}
    # Block 2: j in {2,3}, k in {0,1}
    # Block 3: j in {0,1}, k in {2,3}
    # Block 4: j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0), # j_offset, k_offset
        (2, 0),
        (0, 2),
        (2, 2)
    ]
    
    for b_idx, (j_off, k_off) in enumerate(blocks):
        r_start = b_idx * 11
        r_end = r_start + 11
        
        for i in range(3):
            for j_loc in range(2):
                j_glob = j_loc + j_off
                u_idx = i * m + j_glob
                U[u_idx, r_start:r_end] += U_loc[i * 2 + j_loc, :]
                
        for j_loc in range(2):
            j_glob = j_loc + j_off
            for k_loc in range(2):
                k_glob = k_loc + k_off
                v_idx = j_glob * p + k_glob
                V[v_idx, r_start:r_end] += V_loc[j_loc * 2 + k_loc, :]
                
        for k_loc in range(2):
            k_glob = k_loc + k_off
            for i in range(3):
                w_idx = k_glob * n + i
                W[w_idx, r_start:r_end] += W_loc[k_loc * 3 + i, :]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
