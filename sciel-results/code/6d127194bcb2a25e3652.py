import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 coefficients
    # U_in: 4x7, V_in: 4x7, W_in: 4x7
    # Indices: 0->11, 1->12, 2->21, 3->22
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Outer Strassen coefficients for blocks
    # C_outer[k1, b] where b in 0..3 (11, 12, 21, 22)
    C_outer = np.zeros((7, 4), dtype=int)
    D_outer = np.zeros((7, 4), dtype=int)
    
    # k=1: A11+A22, B11+B22
    C_outer[0, 0] = 1; C_outer[0, 3] = 1
    D_outer[0, 0] = 1; D_outer[0, 3] = 1
    # k=2: A21+A22, B11
    C_outer[1, 2] = 1; C_outer[1, 3] = 1
    D_outer[1, 0] = 1
    # k=3: A11, B12-B22
    C_outer[2, 0] = 1
    D_outer[2, 1] = 1; D_outer[2, 3] = -1
    # k=4: A22, B21-B11
    C_outer[3, 3] = 1
    D_outer[3, 2] = 1; D_outer[3, 0] = -1
    # k=5: A11+A12, B22
    C_outer[4, 0] = 1; C_outer[4, 1] = 1
    D_outer[4, 3] = 1
    # k=6: A21-A11, B11+B12
    C_outer[5, 2] = 1; C_outer[5, 0] = -1
    D_outer[5, 0] = 1; D_outer[5, 1] = 1
    # k=7: A12-A22, B21+B22
    C_outer[6, 1] = 1; C_outer[6, 3] = -1
    D_outer[6, 2] = 1; D_outer[6, 3] = 1
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for k1 in range(7):
        for k2 in range(7):
            c_idx = k1 * 7 + k2
            
            # Fill U and V
            for r in range(16):
                i = r // 4
                j = r % 4
                br = i // 2
                bc = j // 2
                b = br * 2 + bc
                u = i % 2
                v = j % 2
                local_a = u * 2 + v
                
                U[r, c_idx] = S_U[local_a, k2] * C_outer[k1, b]
                V[r, c_idx] = S_V[local_a, k2] * D_outer[k1, b]
                
            # Fill W
            for r in range(16):
                i = r // 4
                j = r % 4
                br = i // 2
                bc = j // 2
                b_c = br * 2 + bc
                u_c = i % 2
                v_c = j % 2
                local_m = u_c * 2 + v_c
                
                W[r, c_idx] = S_W[b_c, k1] * S_W[local_m, k2]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
