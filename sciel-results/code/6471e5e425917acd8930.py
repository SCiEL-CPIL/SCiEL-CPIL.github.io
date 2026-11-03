import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, signs_u, signs_v, signs_w):
        nonlocal col
        for idx, val in zip(u_indices, signs_u):
            U[idx][col] = val
        for idx, val in zip(v_indices, signs_v):
            V[idx][col] = val
        for idx, val in zip(w_indices, signs_w):
            W[idx][col] = val
        col += 1

    for t in range(5):
        # B columns: 2t, 2t+1
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # V indices
        v_00 = 10 * 0 + k0
        v_01 = 10 * 0 + k1
        v_10 = 10 * 1 + k0
        v_11 = 10 * 1 + k1
        v_20 = 10 * 2 + k0
        v_21 = 10 * 2 + k1
        
        # U indices
        u_00 = 0
        u_01 = 1
        u_02 = 2
        u_10 = 3
        u_11 = 4
        u_12 = 5
        
        # W indices
        w_00 = 2 * k0 + 0
        w_01 = 2 * k1 + 0
        w_10 = 2 * k0 + 1
        w_11 = 2 * k1 + 1
        
        # 1. m1 = (a00 + a11) * (b00 + b11)
        add_term([u_00, u_11], [v_00, v_11], [w_00, w_11], [1, 1], [1, 1], [1, 1])
        
        # 2. m2 = (a10 + a11) * b00
        add_term([u_10, u_11], [v_00], [w_10, w_11], [1, 1], [1], [1, -1])
        
        # 3. m3 = a00 * (b01 - b11)
        add_term([u_00], [v_01, v_11], [w_01, w_11], [1], [1, -1], [1, 1])
        
        # 4. m4 = a11 * (b10 - b01)
        add_term([u_11], [v_10, v_01], [w_00, w_10], [1], [1, -1], [1, 1])
        
        # 5. m5 = (a00 + a01) * b11
        add_term([u_00, u_01], [v_11], [w_00, w_01], [1, 1], [1], [-1, 1])
        
        # 6. m6 = (a10 - a00) * (b00 + b01)
        add_term([u_10, u_00], [v_00, v_01], [w_11], [1, -1], [1, 1], [1])
        
        # 7. m7 = (a01 - a11) * (b10 + b11)
        add_term([u_01, u_11], [v_10, v_11], [w_00], [1, -1], [1, 1], [1])
        
        # 8. m8 = a02 * b20
        add_term([u_02], [v_20], [w_00], [1], [1], [1])
        
        # 9. m9 = a02 * b21
        add_term([u_02], [v_21], [w_01], [1], [1], [1])
        
        # 10. m10 = a12 * b20
        add_term([u_12], [v_20], [w_10], [1], [1], [1])
        
        # 11. m11 = a12 * b21
        add_term([u_12], [v_21], [w_11], [1], [1], [1])
        
    return {"U": U, "V": V, "W": W}
