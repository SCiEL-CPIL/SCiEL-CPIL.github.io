import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin for 150s limit
    
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize flat arrays for U, V, W
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U_flat = [0] * (8 * rank)
    V_flat = [0] * (16 * rank)
    W_flat = [0] * (8 * rank)
    
    def set_u(i, j, k):
        U_flat[i * rank + k] += j
        
    def set_v(i, j, k):
        V_flat[i * rank + k] += j
        
    def set_w(i, j, k):
        W_flat[i * rank + k] += j
        
    # Indexing helpers
    # U index: row (0..1) * 4 + col (0..3)
    def u_idx(r, c): return r * 4 + c
    # V index: row (0..3) * 4 + col (0..3)
    def v_idx(r, c): return r * 4 + c
    # W index: col (0..3) * 2 + row (0..1) -> matches W[k*n+i]
    def w_idx(r, c): return c * 2 + r
    
    current_rank = 0
    
    # Define the 4 blocks of 2x2 multiplications
    # (A_row_start, A_col_start, B_row_start, B_col_start, C_col_start)
    # Note: A_row_start is always 0 since A is 2x4, we use rows 0-1.
    # The blocks correspond to:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    blocks = [
        (0, 0, 0, 0, 0), # Block 1
        (0, 2, 2, 0, 0), # Block 2
        (0, 0, 0, 2, 2), # Block 3
        (0, 2, 2, 2, 2)  # Block 4
    ]
    
    for (ar, ac, br, bc, cc) in blocks:
        # Local indices for 2x2 sub-matrices
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
        
        # Strassen's 7 multiplications for 2x2 block
        # M1 = (A11 + A22)(B11 + B22)
        set_u(u11, 1, current_rank)
        set_u(u22, 1, current_rank)
        set_v(v11, 1, current_rank)
        set_v(v22, 1, current_rank)
        set_w(w11, 1, current_rank)
        set_w(w22, 1, current_rank)
        current_rank += 1
        
        # M2 = (A21 + A22)B11
        set_u(u21, 1, current_rank)
        set_u(u22, 1, current_rank)
        set_v(v11, 1, current_rank)
        set_w(w21, 1, current_rank)
        set_w(w22, -1, current_rank)
        current_rank += 1
        
        # M3 = A11(B12 - B22)
        set_u(u11, 1, current_rank)
        set_v(v12, 1, current_rank)
        set_v(v22, -1, current_rank)
        set_w(w12, 1, current_rank)
        set_w(w22, 1, current_rank)
        current_rank += 1
        
        # M4 = A22(B21 - B11)
        set_u(u22, 1, current_rank)
        set_v(v21, 1, current_rank)
        set_v(v11, -1, current_rank)
        set_w(w11, 1, current_rank)
        set_w(w21, 1, current_rank)
        current_rank += 1
        
        # M5 = (A11 + A12)B22
        set_u(u11, 1, current_rank)
        set_u(u12, 1, current_rank)
        set_v(v22, 1, current_rank)
        set_w(w11, -1, current_rank)
        set_w(w12, 1, current_rank)
        current_rank += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        set_u(u21, 1, current_rank)
        set_u(u11, -1, current_rank)
        set_v(v11, 1, current_rank)
        set_v(v12, 1, current_rank)
        set_w(w22, 1, current_rank)
        current_rank += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        set_u(u12, 1, current_rank)
        set_u(u22, -1, current_rank)
        set_v(v21, 1, current_rank)
        set_v(v22, 1, current_rank)
        set_w(w11, 1, current_rank)
        current_rank += 1
        
        if time.time() > deadline:
            break

    # Convert flat lists to list of lists
    U = [U_flat[i*rank:(i+1)*rank] for i in range(8)]
    V = [V_flat[i*rank:(i+1)*rank] for i in range(16)]
    W = [W_flat[i*rank:(i+1)*rank] for i in range(8)]
    
    return {"U": U, "V": V, "W": W}
