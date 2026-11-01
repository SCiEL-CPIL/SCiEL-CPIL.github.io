import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7 products
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11 (local row-major)
    
    # M1
    s1_A = {0: 1, 3: 1}
    s1_B = {0: 1, 3: 1}
    s1_C = {0: 1, 3: 1} # M1 contributes to C00 and C11?
    # Wait, C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's list contributions to C_local
    # C00 (idx 0): M1(+), M4(+), M5(-), M7(+)
    # C01 (idx 1): M3(+), M5(+)
    # C10 (idx 2): M2(+), M4(+)
    # C11 (idx 3): M1(+), M2(-), M3(+), M6(+)
    
    strassen_defs = [
        # (A_signs, B_signs, C_signs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2 -> C10(+), C11(-)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3 -> C01(+), C11(+)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4 -> C00(+), C10(+)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5 -> C00(-), C01(+)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6 -> C11(+)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})             # M7 -> C00(+)
    ]
    
    # Note: C_local indices:
    # 0: (0,0)
    # 1: (0,1)
    # 2: (1,0)
    # 3: (1,1)
    
    col = 0
    for jb in range(2): # 0, 1
        for kb in range(3): # 0, 1, 2
            for r_idx in range(7):
                r = col + r_idx
                A_signs, B_signs, C_signs = strassen_defs[r_idx]
                
                # Fill U (A part)
                for loc_idx, sign in A_signs.items():
                    li, lj = divmod(loc_idx, 2) # loc_idx = i*2 + j'
                    # Global i = li, Global j = jb*2 + lj
                    a = li * m + (jb * 2 + lj)
                    U[a, r] = sign
                    
                # Fill V (B part)
                for loc_idx, sign in B_signs.items():
                    lj, lk = divmod(loc_idx, 2) # loc_idx = j'*2 + l
                    # Global j = jb*2 + lj, Global k = kb*2 + lk
                    b = (jb * 2 + lj) * p + (kb * 2 + lk)
                    V[b, r] = sign
                    
                # Fill W (C part)
                for loc_idx, sign in C_signs.items():
                    li, ll = divmod(loc_idx, 2) # loc_idx = i*2 + l
                    # Global i = li, Global k = kb*2 + ll
                    # W index = k*n + i = (kb*2+ll)*2 + li
                    c = (kb * 2 + ll) * n + li
                    W[c, r] = sign
                
            col += 7
            
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
