import numpy as np

def construct():
    """
    Decompose 4x4 matrix multiplication using recursive Strassen.
    
    Rank = 49.
    """
    # Strassen factors for 2x2 matrix multiplication
    # U, V, W are 4x7 matrices
    # Indices 0..3 correspond to (0,0), (0,1), (1,0), (1,1)
    
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
    
    col = 0
    for outer_term in range(7):
        for inner_term in range(7):
            # Combine outer and inner factors
            # U_global[global_idx, col] += U_outer[outer_idx] * U_inner[inner_idx]
            # global_idx = (ob_r * 2 + in_r) * 4 + (ob_c * 2 + in_c)
            
            for outer_idx in range(4):
                u_val = U_2x2[outer_idx, outer_term]
                if u_val == 0: continue
                ob_r = outer_idx // 2
                ob_c = outer_idx % 2
                
                for inner_idx in range(4):
                    u_val_in = U_2x2[inner_idx, inner_term]
                    if u_val_in == 0: continue
                    in_r = inner_idx // 2
                    in_c = inner_idx % 2
                    
                    gr = ob_r * 2 + in_r
                    gc = ob_c * 2 + in_c
                    U[gr * 4 + gc, col] += u_val * u_val_in
                    
            for outer_idx in range(4):
                v_val_out = V_2x2[outer_idx, outer_term]
                if v_val_out == 0: continue
                ob_r = outer_idx // 2
                ob_c = outer_idx % 2
                
                for inner_idx in range(4):
                    v_val_in = V_2x2[inner_idx, inner_term]
                    if v_val_in == 0: continue
                    in_r = inner_idx // 2
                    in_c = inner_idx % 2
                    
                    gr = ob_r * 2 + in_r
                    gc = ob_c * 2 + in_c
                    V[gr * 4 + gc, col] += v_val_out * v_val_in
                    
            for outer_idx in range(4):
                w_val_out = W_2x2[outer_idx, outer_term]
                if w_val_out == 0: continue
                ob_r = outer_idx // 2
                ob_c = outer_idx % 2
                
                for inner_idx in range(4):
                    w_val_in = W_2x2[inner_idx, inner_term]
                    if w_val_in == 0: continue
                    in_r = inner_idx // 2
                    in_c = inner_idx % 2
                    
                    gr = ob_r * 2 + in_r
                    gc = ob_c * 2 + in_c
                    # W index: gc * 4 + gr (column-major flattening as per previous valid solution)
                    W[gc * 4 + gr, col] += w_val_out * w_val_in
                    
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
