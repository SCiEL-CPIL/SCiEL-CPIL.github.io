import time
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = time.time() + 140  # 140s deadline, 20s safety margin from 160s limit

def construct():
    n, m, p = 3, 4, 6
    
    # We construct a valid Strassen-based decomposition.
    # The current program has a score of 1.0909, which implies a rank of 66 (since 72/66 = 1.0909).
    # We will verify this rank is achievable and return the decomposition.
    
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    curr_rank = 0
    
    def add_mults(u_specs, v_specs, w_specs):
        nonlocal curr_rank
        if curr_rank >= rank:
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

    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen Coefficients for 2x2x2
    # Local A: [[a0, a1], [a2, a3]]
    # Local B: [[b0, b1], [b2, b3]]
    # Local C: [[c0, c1], [c2, c3]]
    
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2-a0)(b0+b1)
    # M7 = (a1-a3)(b2+b3)
    
    # C reconstruction:
    # c0 = M1 - M5 + M7
    # c1 = M3 + M5
    # c2 = M2 + M4
    # c3 = M1 - M2 + M6
    
    s_defs = [
        # (a_coeffs, b_coeffs, c_coeffs)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # M3
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # M7
    ]

    # Loop over the 3 blocks of P (columns of C)
    for k in range(3):
        # Base indices for B and C columns
        b_col_base = 2 * k
        c_col_base = 2 * k
        
        # 1. Schoolbook part for A[0] (1x4)
        # 8 mults
        for j in range(4):
            for c_loc in range(2):
                # A[0, j] * B[j, 2k + c_loc] -> C[0, 2k + c_loc]
                u_spec = [(u_idx(0, j), 1)]
                v_spec = [(v_idx(j, b_col_base + c_loc), 1)]
                w_spec = [(w_idx(b_col_base + c_loc, 0), 1)]
                add_mults(u_spec, v_spec, w_spec)

        # 2. Strassen part for A[1:3] (2x4)
        # Split into Left (cols 0,1) and Right (cols 2,3) of A
        # Left: A[1:3, 0:2], Right: A[1:3, 2:4]
        
        a_row_off = 1
        b_row_off_l = 0
        b_row_off_r = 2
        
        # Left Block: A[1:3, 0:2] * B[0:2, 2k:2k+2]
        def get_a_l_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c)
            
        def get_b_l_idx(loc, row_off):
            r, c = divmod(loc, 2)
            return v_idx(row_off + r, b_col_base + c)
            
        # C local output maps to C[1:3, 2k:2k+2]
        def get_c_l_idx(loc):
            r, c = divmod(loc, 2)
            return w_idx(b_col_base + c, a_row_off + r)

        for a_c, b_c, c_c in s_defs:
            u_spec = [(get_a_l_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_l_idx(i, b_row_off_l), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_l_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

        # Right Block: A[1:3, 2:4] * B[2:4, 2k:2k+2]
        def get_a_r_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c + 2)
            
        for a_c, b_c, c_c in s_defs:
            u_spec = [(get_a_r_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_l_idx(i, b_row_off_r), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_l_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

    # Verification step (optional, but good for debugging in development)
    # In production, we just return.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
