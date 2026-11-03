import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # We construct a rank-56 decomposition using 8 independent Strassen 2x2 decompositions.
    # This is a known upper bound.
    
    # Strassen 2x2 coefficients
    # U_loc: 4x7, V_loc: 4x7, W_loc: 4x7
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))

    # M1 = (a00+a11)(b00+b11)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # M3 = a00(b01-b11)
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # M4 = a11(b10-b00)
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # M5 = (a00+a01)b11
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    total_rank = 56
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Mapping logic for 2x4 * 4x8
    # Split A (2x4) into 2 blocks of 2x2 along columns.
    # Split B (4x8) into 8 blocks of 2x2 (2 row-blocks x 4 col-blocks).
    # Output C (2x8) has 4 column-pairs.
    
    group = 0
    for jj in range(p // 2):        # 0, 1, 2, 3 (column pair index)
        for half in range(2):       # 0, 1 (A-half / B-row-half)
            base_r = group * 7
            group += 1

            # A-half 'half': columns 2*half, 2*half+1
            # Local A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            # Global A index: i*m + j
            a_map = [
                0 * m + (2 * half + 0),
                0 * m + (2 * half + 1),
                1 * m + (2 * half + 0),
                1 * m + (2 * half + 1),
            ]
            
            # B-half 'half': rows 2*half, 2*half+1; cols 2*jj, 2*jj+1
            # Local B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            # Global B index: j*p + k
            b_map = [
                (2 * half + 0) * p + (2 * jj + 0),
                (2 * half + 0) * p + (2 * jj + 1),
                (2 * half + 1) * p + (2 * jj + 0),
                (2 * half + 1) * p + (2 * jj + 1),
            ]
            
            # C local: c00->(i=0, k=2jj), c01->(i=0, k=2jj+1), c10->(i=1, k=2jj), c11->(i=1, k=2jj+1)
            # W index convention: k*n + i
            c_map = [
                (2 * jj + 0) * n + 0,
                (2 * jj + 1) * n + 0,
                (2 * jj + 0) * n + 1,
                (2 * jj + 1) * n + 1,
            ]

            # Fill U, V, W for this group
            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la, r]
                    if val != 0:
                        U[a_map[la], rr] += val
                for lb in range(4):
                    val = V_loc[lb, r]
                    if val != 0:
                        V[b_map[lb], rr] += val
                for lc in range(4):
                    val = W_loc[lc, r]
                    if val != 0:
                        W[c_map[lc], rr] += val

    # Safety deadline guard
    if time.time() - start_time > 115:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
