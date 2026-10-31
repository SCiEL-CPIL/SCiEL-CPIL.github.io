import numpy as np
import time

def construct():
    """
    Decompose 4x4 matrix multiplication using tensor product of 2x2 Strassen.
    
    Rank = 49 = 7 * 7.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    # Strassen factors for 2x2 matrix multiplication
    # Indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U_2x2 = np.zeros((4, 7), dtype=int)
    V_2x2 = np.zeros((4, 7), dtype=int)
    W_2x2 = np.zeros((4, 7), dtype=int)
    
    # Term 1: (A00+A11)(B00+B11)
    U_2x2[0, 0] = 1; U_2x2[3, 0] = 1
    V_2x2[0, 0] = 1; V_2x2[3, 0] = 1
    W_2x2[0, 0] = 1; W_2x2[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U_2x2[2, 1] = 1; U_2x2[3, 1] = 1
    V_2x2[0, 1] = 1
    W_2x2[2, 1] = 1; W_2x2[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U_2x2[0, 2] = 1
    V_2x2[1, 2] = 1; V_2x2[3, 2] = -1
    W_2x2[1, 2] = 1; W_2x2[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U_2x2[3, 3] = 1
    V_2x2[2, 3] = 1; V_2x2[0, 3] = -1
    W_2x2[0, 3] = 1; W_2x2[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    U_2x2[0, 4] = 1; U_2x2[1, 4] = 1
    V_2x2[3, 4] = 1
    W_2x2[0, 4] = -1; W_2x2[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U_2x2[2, 5] = 1; U_2x2[0, 5] = -1
    V_2x2[0, 5] = 1; V_2x2[1, 5] = 1
    W_2x2[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U_2x2[1, 6] = 1; U_2x2[3, 6] = -1
    V_2x2[2, 6] = 1; V_2x2[3, 6] = 1
    W_2x2[0, 6] = 1
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Pre-compute index mappings for 2x2 to 4x4
    # idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # Global row = outer_r * 2 + inner_r
    # Global col = outer_c * 2 + inner_c
    
    for outer_term in range(7):
        for inner_term in range(7):
            col = outer_term * 7 + inner_term
            
            # U factors
            for outer_idx in range(4):
                u_out = U_2x2[outer_idx, outer_term]
                if u_out == 0: continue
                or_ = outer_idx // 2
                oc = outer_idx % 2
                for inner_idx in range(4):
                    u_in = U_2x2[inner_idx, inner_term]
                    if u_in == 0: continue
                    ir = inner_idx // 2
                    ic = inner_idx % 2
                    U[(or_*2+ir)*4 + (oc*2+ic), col] += u_out * u_in

            # V factors
            for outer_idx in range(4):
                v_out = V_2x2[outer_idx, outer_term]
                if v_out == 0: continue
                or_ = outer_idx // 2
                oc = outer_idx % 2
                for inner_idx in range(4):
                    v_in = V_2x2[inner_idx, inner_term]
                    if v_in == 0: continue
                    ir = inner_idx // 2
                    ic = inner_idx % 2
                    V[(or_*2+ir)*4 + (oc*2+ic), col] += v_out * v_in

            # W factors (Note: W index is col*n + row)
            for outer_idx in range(4):
                w_out = W_2x2[outer_idx, outer_term]
                if w_out == 0: continue
                or_ = outer_idx // 2
                oc = outer_idx % 2
                for inner_idx in range(4):
                    w_in = W_2x2[inner_idx, inner_term]
                    if w_in == 0: continue
                    ir = inner_idx // 2
                    ic = inner_idx % 2
                    W[(oc*2+ic)*4 + (or_*2+ir), col] += w_out * w_in
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
