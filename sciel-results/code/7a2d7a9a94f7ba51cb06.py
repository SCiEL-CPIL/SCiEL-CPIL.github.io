"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Define Strassen terms for 2x2x2 as (U_vec, V_vec, W_vec)
    # Indices: U is (i,j), V is (j,k), W is (k,i)
    # We represent sparse vectors as lists of (index, value)
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Strassen 7 terms
    # 1. (A00+A11)(B00+B11)
    t1_u = [(idx_u(0,0), 1), (idx_u(1,1), 1)]
    t1_v = [(idx_v(0,0), 1), (idx_v(1,1), 1)]
    t1_w = [(idx_w(0,0), 1), (idx_w(1,1), 1)]
    
    # 2. (A10+A11)B00
    t2_u = [(idx_u(1,0), 1), (idx_u(1,1), 1)]
    t2_v = [(idx_v(0,0), 1)]
    t2_w = [(idx_w(0,1), 1), (idx_w(1,1), -1)]
    
    # 3. A00(B01-B11)
    t3_u = [(idx_u(0,0), 1)]
    t3_v = [(idx_v(0,1), 1), (idx_v(1,1), -1)]
    t3_w = [(idx_w(1,0), 1), (idx_w(1,1), 1)]
    
    # 4. A11(B10-B00)
    t4_u = [(idx_u(1,1), 1)]
    t4_v = [(idx_v(1,0), 1), (idx_v(0,0), -1)]
    t4_w = [(idx_w(0,0), 1), (idx_w(0,1), 1)]
    
    # 5. (A00+A01)B11
    t5_u = [(idx_u(0,0), 1), (idx_u(0,1), 1)]
    t5_v = [(idx_v(1,1), 1)]
    t5_w = [(idx_w(1,0), 1), (idx_w(0,0), -1)]
    
    # 6. (A10-A00)(B00+B01)
    t6_u = [(idx_u(1,0), 1), (idx_u(0,0), -1)]
    t6_v = [(idx_v(0,0), 1), (idx_v(0,1), 1)]
    t6_w = [(idx_w(1,1), 1)]
    
    # 7. (A01-A11)(B10+B11)
    t7_u = [(idx_u(0,1), 1), (idx_u(1,1), -1)]
    t7_v = [(idx_v(1,0), 1), (idx_v(1,1), 1)]
    t7_w = [(idx_w(0,0), 1)]
    
    terms = [
        (t1_u, t1_v, t1_w),
        (t2_u, t2_v, t2_w),
        (t3_u, t3_v, t3_w),
        (t4_u, t4_v, t4_w),
        (t5_u, t5_v, t5_w),
        (t6_u, t6_v, t6_w),
        (t7_u, t7_v, t7_w)
    ]
    
    # Fill matrices for 2 blocks
    for r in range(7):
        u_vec, v_vec, w_vec = terms[r]
        
        # Block 1: columns 0, 1 of B
        col = r
        for idx, val in u_vec:
            U[idx][col] += val
        for idx, val in v_vec:
            V[idx][col] += val
        for idx, val in w_vec:
            W[idx][col] += val
            
        # Block 2: columns 2, 3 of B
        # This corresponds to shifting k in V and W by 2
        col = r + 7
        for idx, val in u_vec:
            U[idx][col] += val
            
        # Shift V indices: j*p + k -> j*p + (k+2)
        # Since p=4, indices are 0,1 (k=0,1) and 2,3 (k=2,3).
        # The term indices in v_vec are for k=0,1.
        # We need to map them to k=2,3.
        # idx_v(j,k) = j*4 + k.
        # New idx = j*4 + (k+2) = idx + 2.
        for idx, val in v_vec:
            V[idx + 2][col] += val
            
        # Shift W indices: k*n + i -> (k+2)*n + i
        # idx_w(k,i) = k*2 + i.
        # New idx = (k+2)*2 + i = k*2 + i + 4 = idx + 4.
        for idx, val in w_vec:
            W[idx + 4][col] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
