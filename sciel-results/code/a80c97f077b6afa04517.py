import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # We decompose into two 2x2x8 multiplications
    # Rank of 2x2x8 is 28 (4 blocks of 2x2x2 Strassen, rank 7 each)
    # Total rank 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # Indices i,j in 0..1 for U (2x2)
    # Indices j,k in 0..1 for V (2x2)
    # Indices k,i in 0..1 for W (2x2)
    
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U (coefficients for A)
    # U is 4x7 (flattened 2x2)
    U_strassen = [
        [(0,1), (3,1)],    # M1: A00+A11
        [(2,1), (3,1)],    # M2: A10+A11
        [(0,1)],           # M3: A00
        [(3,1)],           # M4: A11
        [(0,1), (1,1)],    # M5: A00+A01
        [(2,1), (0,-1)],   # M6: A10-A00
        [(1,1), (3,-1)]    # M7: A01-A11
    ]
    
    # V (coefficients for B)
    # V is 4x7 (flattened 2x2)
    V_strassen = [
        [(0,1), (3,1)],    # M1: B00+B11
        [(0,1)],           # M2: B00
        [(2,1), (3,-1)],   # M3: B10-B11
        [(1,1), (0,-1)],   # M4: B01-B00
        [(3,1)],           # M5: B11
        [(0,1), (2,1)],    # M6: B00+B10
        [(1,1), (3,1)]     # M7: B01+B11
    ]
    
    # W (coefficients for C)
    # W is 4x7 (flattened 2x2)
    # C00 = M1 - M4 + M5 + M6
    # C01 = M4 + M5
    # C10 = M2 + M3 - M6 - M7
    # C11 = M1 - M3 + M2 + M7
    # Note: W index is k*n + i. n=2.
    # k=0,i=0 -> idx 0
    # k=0,i=1 -> idx 1
    # k=1,i=0 -> idx 2
    # k=1,i=1 -> idx 3
    
    W_strassen = [
        [(0,1), (2,1)],    # M1: C00 (k=0,i=0) NO, W is for C.
                           # C00 uses M1, M4, M5, M6
                           # Let's map M_r to C entries.
        [(1,1), (3,1)],    # M2?
        [(2,1), (3,1)],    # M3?
        [(0,1), (1,1)],    # M4?
        [(0,1), (3,1)],    # M5?
        [(0,-1), (2,1)],   # M6?
        [(1,1), (3,1)]     # M7?
    ]
    
    # Let's re-verify W mapping for Strassen 2x2x2
    # C00 = M1 - M4 + M5 + M6
    # C01 = M4 + M5
    # C10 = M2 + M3 - M6 - M7
    # C11 = M1 - M3 + M2 + M7
    
    # W[r] contains contributions of M_r to C.
    # W is flattened C (2x2) -> 4 entries.
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    W_strassen = [
        [(0,1), (3,1)],    # M1: C00 (+1), C11 (+1)
        [(2,1), (3,1)],    # M2: C10 (+1), C11 (+1)
        [(2,1), (3,-1)],   # M3: C10 (+1), C11 (-1)
        [(0,-1), (1,1)],   # M4: C00 (-1), C01 (+1)
        [(0,1), (1,1)],    # M5: C00 (+1), C01 (+1)
        [(0,1), (2,-1)],   # M6: C00 (+1), C10 (-1)
        [(2,-1), (3,1)]    # M7: C10 (-1), C11 (+1)
    ]
    
    # Assemble 2x2x8 blocks
    # We have 4 blocks of 2x2x2 for the 2x2x8 part.
    # Block t (0..3) corresponds to B columns 2t, 2t+1.
    
    # Helper to add a Strassen block
    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            idx_r = base_rank + r
            # U
            for u_idx, val in U_strassen[r]:
                ui, uj = divmod(u_idx, 2)
                gi = ui + i_off
                gj = uj + j_off
                U[gi*4 + gj][idx_r] += val
            
            # V
            for v_idx, val in V_strassen[r]:
                vj, vk = divmod(v_idx, 2)
                gj = vj + j_off
                gk = vk + k_off
                V[gj*8 + gk][idx_r] += val
            
            # W
            for w_idx, val in W_strassen[r]:
                wk, wi = divmod(w_idx, 2)
                gk = wk + k_off
                gi = wi + i_off
                W[gk*2 + gi][idx_r] += val

    # Part 1: A_L (i:0-1, j:0-1) * B_L (j:0-1, k:0-7)
    # B_L is 2x8, split into 4 2x2 blocks.
    for t in range(4):
        add_strassen_block(t*7, 0, 0, t*2)
        
    # Part 2: A_R (i:0-1, j:2-3) * B_R (j:2-3, k:0-7)
    # B_R is 2x8, split into 4 2x2 blocks.
    for t in range(4):
        add_strassen_block(28 + t*7, 0, 2, t*2)
        
    return {"U": U, "V": V, "W": W}
