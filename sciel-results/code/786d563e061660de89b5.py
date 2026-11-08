import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Base case: Strassen decomposition for 2x2 matrix multiplication (Rank 7)
    # u: coeffs for A (2x2), v: coeffs for B (2x2), w: coeffs for C (2x2)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    base_u = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    base_v = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [-1, 0, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    base_w = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 0, 1, -1], # M2: C10 - C11
        [0, 1, 0, 1], # M3: C01 + C11
        [1, 0, 1, 0], # M4: C00 + C10
        [-1, 1, 0, 0], # M5: -C00 + C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Step 1: Build 4x4x4 decomposition (Rank 49) using Kronecker products
    # u_4x4 = u_out (Kron) u_in
    # v_4x4 = v_out (Kron) v_in
    # w_4x4 = w_out (Kron) w_in
    
    u_4x4 = []
    v_4x4 = []
    w_4x4 = []
    
    for s in range(7):
        for t in range(7):
            u_new = np.kron(base_u[s], base_u[t])
            v_new = np.kron(base_v[s], base_v[t])
            w_new = np.kron(base_w[s], base_w[t])
            u_4x4.append(u_new)
            v_4x4.append(v_new)
            w_4x4.append(w_new)
            
    rank_4x4 = 49
    total_rank = 98
    
    # Initialize matrices for 4x4x8
    # U: (n*m, rank) = (16, 98)
    # V: (m*p, rank) = (32, 98)
    # W: (p*n, rank) = (32, 98)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill first 49 terms (for C_L, k in 0..3)
    for r in range(rank_4x4):
        u_vec = u_4x4[r]
        v_vec = v_4x4[r]
        w_vec = w_4x4[r]
        
        # U: indices i*m + j. u_vec is length 16.
        for idx in range(16):
            U[idx][r] = u_vec[idx]
            
        # V: indices j*p + k. v_vec is length 16 (for 4x4 B).
        # Map v_vec[j*4 + k] to V[j*8 + k] for k in 0..3
        for j in range(4):
            for k in range(4):
                V[j * 8 + k][r] = v_vec[j * 4 + k]
                
        # W: indices k*n + i. w_vec is length 16 (for 4x4 C).
        # Map w_vec[k*4 + i] to W[k*4 + i] for k in 0..3
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][r] = w_vec[k * 4 + i]
                
    # Fill next 49 terms (for C_R, k in 4..7)
    for r in range(rank_4x4):
        u_vec = u_4x4[r]
        v_vec = v_4x4[r]
        w_vec = w_4x4[r]
        
        col = r + rank_4x4
        
        # U: same as before
        for idx in range(16):
            U[idx][col] = u_vec[idx]
            
        # V: Map v_vec[j*4 + k] to V[j*8 + (k+4)]
        for j in range(4):
            for k in range(4):
                V[j * 8 + (k + 4)][col] = v_vec[j * 4 + k]
                
        # W: Map w_vec[k*4 + i] to W[(k+4)*4 + i]
        for k in range(4):
            for i in range(4):
                W[(k + 4) * 4 + i][col] = w_vec[k * 4 + i]
                
    return {"U": U, "V": V, "W": W}
