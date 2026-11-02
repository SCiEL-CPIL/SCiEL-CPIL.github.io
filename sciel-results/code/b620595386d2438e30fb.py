import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 4
    rank = 28
    
    # We will build the matrices as flat arrays first for easier index management
    # U is 8x28, V is 16x28, W is 8x28
    U_flat = [0] * (8 * rank)
    V_flat = [0] * (16 * rank)
    W_flat = [0] * (8 * rank)
    
    def set_u(i, j, k):
        U_flat[i * rank + k] += j
        
    def set_v(i, j, k):
        V_flat[i * rank + k] += j
        
    def set_w(i, j, k):
        W_flat[i * rank + k] += j
        
    # u_idx: i (row 0-1), j (col 0-3) -> i*4+j
    # v_idx: i (row 0-3), j (col 0-3) -> i*4+j
    # w_idx: i (row 0-3), j (col 0-1) -> j*2+i (Trace convention: W[k*n+i, r] where k is col of C, i is row of C)
    # C is 2x4. Row r, Col c. Index in W is c*n + r = c*2 + r.
    
    def u_idx(r, c): return r * 4 + c
    def v_idx(r, c): return r * 4 + c
    def w_idx(r, c): return c * 2 + r
    
    current_rank = 0
    
    # 4 blocks
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    blocks = [
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 2, 2, 2, 2)
    ]
    
    for (ar, ac, br, bc, cc) in blocks:
        # Indices
        u11 = u_idx(ar, ac)
        u12 = u_idx(ar, ac + 1)
        u21 = u_idx(ar + 1, ac)
        u22 = u_idx(ar + 1, ac + 1)
        
        v11 = v_idx(br, bc)
        v12 = v_idx(br, bc + 1)
        v21 = v_idx(br + 1, bc)
        v22 = v_idx(br + 1, bc + 1)
        
        w11 = w_idx(0, cc)
        w12 = w_idx(0, cc + 1)
        w21 = w_idx(1, cc)
        w22 = w_idx(1, cc + 1)
        
        # Strassen 7 terms
        
        # M1
        set_u(u11, 1, current_rank)
        set_u(u22, 1, current_rank)
        set_v(v11, 1, current_rank)
        set_v(v22, 1, current_rank)
        set_w(w11, 1, current_rank)
        set_w(w22, 1, current_rank)
        current_rank += 1
        
        # M2
        set_u(u21, 1, current_rank)
        set_u(u22, 1, current_rank)
        set_v(v11, 1, current_rank)
        set_w(w21, 1, current_rank)
        set_w(w22, -1, current_rank)
        current_rank += 1
        
        # M3
        set_u(u11, 1, current_rank)
        set_v(v12, 1, current_rank)
        set_v(v22, -1, current_rank)
        set_w(w12, 1, current_rank)
        set_w(w22, 1, current_rank)
        current_rank += 1
        
        # M4
        set_u(u22, 1, current_rank)
        set_v(v21, 1, current_rank)
        set_v(v11, -1, current_rank)
        set_w(w11, 1, current_rank)
        set_w(w21, 1, current_rank)
        current_rank += 1
        
        # M5
        set_u(u11, 1, current_rank)
        set_u(u12, 1, current_rank)
        set_v(v22, 1, current_rank)
        set_w(w11, -1, current_rank)
        set_w(w12, 1, current_rank)
        current_rank += 1
        
        # M6
        set_u(u21, 1, current_rank)
        set_u(u11, -1, current_rank)
        set_v(v11, 1, current_rank)
        set_v(v12, 1, current_rank)
        set_w(w22, 1, current_rank)
        current_rank += 1
        
        # M7
        set_u(u12, 1, current_rank)
        set_u(u22, -1, current_rank)
        set_v(v21, 1, current_rank)
        set_v(v22, 1, current_rank)
        set_w(w11, 1, current_rank)
        current_rank += 1
        
    if current_rank != rank:
        raise ValueError("Rank mismatch")
        
    # Reshape to 2D lists
    U = [U_flat[i*rank:(i+1)*rank] for i in range(8)]
    V = [V_flat[i*rank:(i+1)*rank] for i in range(16)]
    W = [W_flat[i*rank:(i+1)*rank] for i in range(8)]
    
    return {"U": U, "V": V, "W": W}
