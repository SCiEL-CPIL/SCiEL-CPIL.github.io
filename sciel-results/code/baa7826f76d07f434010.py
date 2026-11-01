import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7 products for 2x2 matrix multiplication
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each entry: (A_indices_with_signs, B_indices_with_signs, C_indices_with_signs)
    # Local 2x2 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_defs = [
        # M1 = (A00+A11)*(B00+B11) -> C00(+), C11(+)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)*B00 -> C10(+), C11(-)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00*(B11-B01) -> C01(+), C11(+)
        ({0: 1}, {1: -1, 3: 1}, {1: 1, 3: 1}),
        # M4 = A11*(B00-B10) -> C00(+), C10(+)
        ({3: 1}, {0: 1, 2: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)*B11 -> C00(-), C01(+)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)*(B00+B01) -> C11(+)
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)*(B10+B11) -> C00(+)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    for jb in range(2):      # B row-block: 0, 1
        for kb in range(3):  # B col-block: 0, 1, 2
            for r_idx in range(7):
                r = col + r_idx
                A_signs, B_signs, C_signs = strassen_defs[r_idx]
                
                # Fill U (A part): A is 2x4, local block is 2x2 at cols [jb*2, jb*2+2)
                for loc_idx, sign in A_signs.items():
                    li, lj = divmod(loc_idx, 2)
                    a = li * m + (jb * 2 + lj)
                    U[a, r] = sign
                    
                # Fill V (B part): B is 4x6, local block is 2x2 at rows [jb*2, jb*2+2), cols [kb*2, kb*2+2)
                for loc_idx, sign in B_signs.items():
                    lj, lk = divmod(loc_idx, 2)
                    b = (jb * 2 + lj) * p + (kb * 2 + lk)
                    V[b, r] = sign
                    
                # Fill W (C part): C is 2x6, local block is 2x2 at cols [kb*2, kb*2+2)
                # W index = k*n + i (trace convention)
                for loc_idx, sign in C_signs.items():
                    li, lk = divmod(loc_idx, 2)
                    c = (kb * 2 + lk) * n + li
                    W[c, r] = sign
                    
            col += 7
            
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
