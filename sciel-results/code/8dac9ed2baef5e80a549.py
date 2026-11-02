import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 3 * 7 = 21
    rank = 21
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # U_part: [A00, A01, A10, A11]
    # V_part: [B00, B01, B10, B11] (indices j*p+k, but for 2x2 block, relative)
    # W_part: [C00, C01, C10, C11] (indices k*n+i, relative)
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A10)B11
    # M6: (A11-A00)(B00+B01)
    # M7: (A00-A10)(B10+B11)
    
    # U vectors for the 7 multiplications
    # Indices in U: 0:00, 1:01, 2:10, 3:11
    U_vecs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 0, 1, 0],  # M5
        [-1, 0, 0, 1], # M6
        [1, 0, -1, 0]  # M7
    ]
    
    # V vectors for the 7 multiplications
    # Indices in V (local 2x2): 0:00, 1:01, 2:10, 3:11
    # Note: V index is j*p + k. For 2x2 block, j=0,1 and k_local=0,1.
    # Global k = 2*block + k_local.
    # V global index = j * 6 + (2*block + k_local).
    V_vecs_local = [
        [1, 0, 0, 1],  # M1: B00 + B11 (j=0,k=0 and j=1,k=1)
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11 (j=0,k=1 and j=1,k=1)
        [0, 0, 1, -1], # M4: B10 - B00 (j=1,k=0 and j=0,k=0) -> Wait.
                       # M4 is A11 * (B10 - B00).
                       # B10 is j=1, k=0. B00 is j=0, k=0.
                       # So V local indices: 2 (10) and 0 (00).
                       # V_vecs_local[3] should be [0, 0, 1, -1] if order is 00,01,10,11.
                       # Let's re-verify V index order.
                       # Local 0: 00, 1: 01, 2: 10, 3: 11.
                       # M1: B00(0) + B11(3) -> [1,0,0,1]
                       # M2: B00(0) -> [1,0,0,0]
                       # M3: B01(1) - B11(3) -> [0,1,0,-1]
                       # M4: B10(2) - B00(0) -> [0,0,1,-1]  <-- Corrected
                       # M5: B11(3) -> [0,0,0,1]
                       # M6: B00(0) + B01(1) -> [1,1,0,0]
                       # M7: B10(2) + B11(3) -> [0,0,1,1]
    ]
    
    # Correction on V_vecs_local
    V_vecs_local = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    # W vectors for the 7 multiplications
    # Indices in W (local 2x2 output): 0:C00, 1:C10, 2:C01, 3:C11?
    # W global index: k * n + i.
    # Local k=0,1. i=0,1.
    # Local 0: k=0,i=0 -> C00
    # Local 1: k=0,i=1 -> C10
    # Local 2: k=1,i=0 -> C01
    # Local 3: k=1,i=1 -> C11
    
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    W_vecs_local = [
        [1, 0, 0, 1],  # M1: C00(1), C11(1)
        [0, 1, 0, -1], # M2: C10(1), C11(-1)
        [0, 0, 1, 1],  # M3: C01(1), C11(1)
        [1, 1, 0, 0],  # M4: C00(1), C10(1)
        [-1, 0, 1, 0], # M5: C00(-1), C01(1)
        [0, 0, 0, 1],  # M6: C11(1)
        [1, 0, 0, 0]   # M7: C00(1)
    ]
    
    # Verify W M5: C00 - M5. C01 + M5.
    # W_vecs_local[4] = [-1, 0, 1, 0]. Correct.
    
    # Verify W M7: C00 + M7.
    # W_vecs_local[6] = [1, 0, 0, 0]. Correct.

    num_blocks = p // 2 # 3 blocks
    
    for block in range(num_blocks):
        offset = block * 7
        for r in range(7):
            col = offset + r
            
            # Fill U
            for i in range(4):
                U[i, col] = U_vecs[r][i]
                
            # Fill V
            # V local index l corresponds to j=l//2, k_local=l%2
            # Global V index = j * p + (2*block + k_local)
            for l in range(4):
                val = V_vecs_local[r][l]
                if val != 0:
                    j = l // 2
                    k_local = l % 2
                    global_v_idx = j * p + (2 * block + k_local)
                    V[global_v_idx, col] = val
                    
            # Fill W
            # W local index l corresponds to k_local=l//2, i=l%2
            # Global W index = (2*block + k_local) * n + i
            for l in range(4):
                val = W_vecs_local[r][l]
                if val != 0:
                    k_local = l // 2
                    i = l % 2
                    global_w_idx = (2 * block + k_local) * n + i
                    W[global_w_idx, col] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
