import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n, m, p = 3, 4, 6
    
    # Rank 66 decomposition using schoolbook for top row and Strassen for bottom
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    curr_rank = 0
    
    def add_mults(u_specs, v_specs, w_specs):
        nonlocal curr_rank
        if curr_rank >= rank:
            return
        if time.time() > deadline:
            return
        for idx, val in u_specs:
            if val != 0:
                U[idx, curr_rank] += val
        for idx, val in v_specs:
            if val != 0:
                V[idx, curr_rank] += val
        for idx, val in w_specs:
            if val != 0:
                W[idx, curr_rank] += val
        curr_rank += 1

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen Coefficients for 2x2x2
    s_defs = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    for k in range(3):
        b_col_base = 2 * k
        c_col_base = 2 * k
        
        # Schoolbook part for A[0] (1x4): 8 mults
        for j in range(4):
            for c_loc in range(2):
                u_spec = [(u_idx(0, j), 1)]
                v_spec = [(v_idx(j, b_col_base + c_loc), 1)]
                w_spec = [(w_idx(b_col_base + c_loc, 0), 1)]
                add_mults(u_spec, v_spec, w_spec)

        # Strassen part for A[1:3] (2x4): 14 mults
        a_row_off = 1
        b_row_off_l = 0
        b_row_off_r = 2
        
        def get_a_l_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c)
            
        def get_b_idx(loc, row_off):
            r, c = divmod(loc, 2)
            return v_idx(row_off + r, b_col_base + c)
            
        def get_c_idx(loc):
            r, c = divmod(loc, 2)
            return w_idx(b_col_base + c, a_row_off + r)

        for a_c, b_c, c_c in s_defs:
            u_spec = [(get_a_l_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_idx(i, b_row_off_l), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

        def get_a_r_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c + 2)
            
        for a_c, b_c, c_c in s_defs:
            u_spec = [(get_a_r_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_idx(i, b_row_off_r), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

    # Verify: check that the decomposition is correct
    # C[i,k] = sum_r U[i*m+j... wait, the convention is W[k*n+i, r]
    # U shape: (n*m, rank) -> U[i*m+j, r]
    # V shape: (m*p, rank) -> V[j*p+k, r]  
    # W shape: (p*n, rank) -> W[k*n+i, r]
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But this needs to hold for all i,j,k. Let's verify numerically.
    
    # Quick sanity check with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for r in range(rank):
                for j in range(m):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            C_decomp[i, k] = s
    
    # The decomposition should satisfy:
    # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But actually the tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # where T_{i,j,k} = 1 if this corresponds to a valid multiplication term
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T_{i,j,k} should be 1 for all i,j,k (the multiplication tensor)
    # And C[i,k] = sum_j A[i,j]*B[j,k] = sum_{j,r} A[i,j]*B[j,k]*T_{i,j,k}
    # = sum_{j,r} A[i,j]*B[j,k]*sum_r U_{i,j,r}*V_{j,k,r}*W_{k,i,r}
    # This doesn't quite work as I wrote it. Let me re-examine the convention.
    
    # The correct interpretation: the multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # (since C[i,k] = sum_j A[i,j] B[j,k])
    # The decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # where W uses the trace convention W[k*n+i, r]
    
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And this must be the ONLY condition (the tensor is all-ones for M_{3,4,6})
    
    # Let's verify this condition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    # If max_err is small, the decomposition is correct
    # For exact integer/half-integer entries, this should be exactly 0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
