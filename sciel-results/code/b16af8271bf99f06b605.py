import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # The rank 42 decomposition based on splitting m=4 into two m=2 blocks
    # and p=6 into three p=2 blocks.
    # Each 2x2x2 block uses Strassen (rank 7).
    # Total blocks: 2 (from m) * 3 (from p) = 6 blocks.
    # Rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen definitions for 2x2x2
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # M1 = (A00+A11)(B00+B11) -> C00(+), C11(+)
    # M2 = (A10+A11)B00 -> C10(+), C11(-)
    # M3 = A00(B10-B11) -> C01(+), C11(+)
    # M4 = A11(B00-B10) -> C00(+), C10(+)
    # M5 = (A00+A01)B11 -> C00(-), C01(+)
    # M6 = (A10-A00)(B00+B01) -> C11(+)
    # M7 = (A01-A11)(B10+B11) -> C00(+)
    
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})             # M7
    ]
    
    col = 0
    # m_split = 2 (splits of m=4)
    # p_split = 3 (splits of p=6)
    for jm in range(2):
        for kp in range(3):
            for r_idx in range(7):
                r = col + r_idx
                A_signs, B_signs, C_signs = strassen_defs[r_idx]
                
                # Fill U (A part)
                # A global index: i * m + j
                # Local A index: i * 2 + j_local
                # j_global = jm * 2 + j_local
                for loc_idx, sign in A_signs.items():
                    li, lj = divmod(loc_idx, 2)
                    a = li * m + (jm * 2 + lj)
                    U[a, r] = sign
                    
                # Fill V (B part)
                # B global index: j * p + k
                # Local B index: j_local * 2 + k_local
                # j_global = jm * 2 + j_local
                # k_global = kp * 2 + k_local
                for loc_idx, sign in B_signs.items():
                    lj, lk = divmod(loc_idx, 2)
                    b = (jm * 2 + lj) * p + (kp * 2 + lk)
                    V[b, r] = sign
                    
                # Fill W (C part)
                # C global index: k * n + i
                # Local C index: i * 2 + k_local
                # k_global = kp * 2 + k_local
                for loc_idx, sign in C_signs.items():
                    li, ll = divmod(loc_idx, 2)
                    c = (kp * 2 + ll) * n + li
                    W[c, r] = sign
                    
            col += 7
            
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
