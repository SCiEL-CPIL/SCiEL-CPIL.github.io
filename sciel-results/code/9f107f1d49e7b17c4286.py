import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local Strassen coefficients for 2x2x2
    # U_L: (i, j) -> coeff
    # V_L: (j, k) -> coeff
    # W_L: (k, i) -> coeff
    # Indices are flattened: i*2+j, j*2+k, k*2+i
    
    strassen_U = [
        [1, 0, 1, 0], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [1, 0, 1, 0], # M6: -A00, A10 (signs handled in V or U? Usually one side)
        [0, 1, 0, 1]  # M7: A01, -A11
    ]
    # Let's be precise with signs.
    # M1: (A00+A11)(B00+B11) -> U: [1,0,0,1], V: [1,0,0,1], W: [1,0,0,1] (C00, C11)
    # M2: (A10+A11)B00 -> U: [0,0,1,1], V: [1,0,0,0], W: [0,0,1,0] (C10)
    # M3: A00(B01-B11) -> U: [1,0,0,0], V: [0,1,0,-1], W: [0,1,0,0] (C01)
    # M4: A11(B10-B00) -> U: [0,0,0,1], V: [0,0,1,-1], W: [0,0,1,0] (C10)
    # M5: (A00+A01)B11 -> U: [1,1,0,0], V: [0,0,0,1], W: [0,1,0,0] (C01)
    # M6: (A10-A00)(B00+B01) -> U: [-1,0,1,0], V: [1,1,0,0], W: [1,0,0,0] (C00)
    # M7: (A01-A11)(B10+B11) -> U: [0,1,0,-1], V: [0,0,1,1], W: [0,0,0,1] (C11)
    
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
    
    # Define blocks: (j_offset, k_offset)
    blocks = [
        (0, 0), # Block 1
        (2, 0), # Block 2
        (0, 2), # Block 3
        (2, 2)  # Block 4
    ]
    
    for b_idx, (j_off, k_off) in enumerate(blocks):
        for s_idx in range(7):
            r = b_idx * 7 + s_idx
            u_vec = U_strassen[s_idx]
            v_vec = V_strassen[s_idx]
            w_vec = W_strassen[s_idx]
            
            # Map local indices to global
            # Local i is 0,1. Global i is 0,1.
            # Local j is 0,1. Global j is j_off + local_j.
            # Local k is 0,1. Global k is k_off + local_k.
            
            for i in range(2):
                for j in range(2):
                    u_idx = i * 4 + (j_off + j)
                    U[u_idx][r] = u_vec[i * 2 + j]
                    
            for j in range(2):
                for k in range(2):
                    v_idx = (j_off + j) * 4 + (k_off + k)
                    V[v_idx][r] = v_vec[j * 2 + k]
                    
            for k in range(2):
                for i in range(2):
                    w_idx = (k_off + k) * 2 + i
                    W[w_idx][r] = w_vec[k * 2 + i]
                    
    return {"U": U, "V": V, "W": W}
