"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    
    # We'll build rank-1 terms as lists of (u_vec, v_vec, w_vec)
    # u_vec: length n*m, v_vec: length m*p, w_vec: length p*n
    terms = []  # list of (u_list, v_list, w_list)
    
    def add_term(u, v, w):
        """u, v, w are lists of length n*m, m*p, p*n respectively"""
        terms.append((list(u), list(v), list(w)))
    
    # Helper: create zero vectors
    def zero_u(): return [0.0] * (n * m)
    def zero_v(): return [0.0] * (m * p)
    def zero_w(): return [0.0] * (p * n)
    
    # Helper: set element in u (A matrix, row i, col j)
    def set_u(u, i, j, val):
        u[i * m + j] = val
    # Helper: set element in v (B matrix, row j, col k)
    def set_v(v, j, k, val):
        v[j * p + k] = val
    # Helper: set element in w (C matrix, row k, col i) -- trace convention W[k*n+i]
    def set_w(w, k, i, val):
        w[k * n + i] = val
    
    # ============================================================
    # PART 1: 4x4x4 multiplication (columns k=0..3 of B)
    # Using Strassen recursion: 7 multiplications of 2x2x2
    # Each 2x2x2 uses Strassen: 7 rank-1 terms
    # Total: 49 terms
    # ============================================================
    
    # The 7 rank-1 terms for 2x2x2 Strassen
    # Each term is (u_2x2, v_2x2, w_2x2) as 4-element vectors
    # u = [A00, A01, A10, A11], v = [B00, B01, B10, B11], w = [C00, C01, C10, C11]
    #
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_2x2 = [
        # (u, v, w) for each of the 7 terms
        # Term 1: M1 = (A00+A11)(B00+B11)
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # w: C00=1, C11=1  (M1 contributes to C00 and C11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        
        # Term 2: M2 = (A10+A11)B00
        # u: A10=1, A11=1
        # v: B00=1
        # w: C10=1  (M2 contributes to C10)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        
        # Term 3: M3 = A00(B01-B11)
        # u: A00=1
        # v: B01=1, B11=-1
        # w: C01=1  (M3 contributes to C01)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        
        # Term 4: M4 = A11(B10-B00)
        # u: A11=1
        # v: B10=1, B00=-1
        # w: C00=1, C10=1  (M4 contributes to C00 and C10)
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        
        # Term 5: M5 = (A00+A01)B11
        # u: A00=1, A01=1
        # v: B11=1
        # w: C00=-1, C01=1  (M5 contributes -1 to C00, +1 to C01)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # u: A10=1, A00=-1
        # v: B00=1, B01=1
        # w: C11=1  (M6 contributes to C11)
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # u: A01=1, A11=-1
        # v: B10=1, B11=1
        # w: C00=1, C11=1  (M7 contributes to C00 and C11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    # Define the outer Strassen coefficients
    # A_r = sum_{block} a_r[block] * A_block
    # B_r = sum_{block} b_r[block] * B_block
    # C_block gets: sum_r c_r[block] * M_r
    
    # Blocks indexed as: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    outer_strassen = [
        # r, A_coeffs [A00,A01,A10,A11], B_coeffs [B00,B01,B10,B11], C_coeffs [C00,C01,C10,C11]
        (1, [1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        (2, [0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2
        (3, [1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # M3
        (4, [0,0,0,1], [0,-1,1,0], [1,0,1,0]),  # M4
        (5, [1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        (6, [0,0,1,-1], [1,1,0,0], [0,0,0,1]),  # M6
        (7, [0,1,0,-1], [0,0,1,1], [1,0,0,1]),  # M7
    ]
    
    # Block offsets in 4x4 matrix:
    # Block 0 (0,0): rows 0-1, cols 0-1
    # Block 1 (0,1): rows 0-1, cols 2-3
    # Block 2 (1,0): rows 2-3, cols 0-1
    # Block 3 (1,1): rows 2-3, cols 2-3
    block_row_start = [0, 0, 2, 2]
    block_col_start = [0, 2, 0, 2]
    
    # For 4x4x4, B is columns 0-3
    # For each outer term r and inner term s:
    for r_idx, (r, a_coeffs, b_coeffs, c_coeffs) in enumerate(outer_strassen):
        for s_idx, (u_s, v_s, w_s) in enumerate(strassen_2x2):
            u = zero_u()
            v = zero_v()
            w = zero_w()
            
            # u: A_r is 4x4, formed from A blocks
            for block in range(4):
                if a_coeffs[block] != 0:
                    br = block_row_start[block]
                    bc = block_col_start[block]
                    for li in range(2):
                        for lj in range(2):
                            gi = br + li
                            gj = bc + lj
                            u[gi * m + gj] += u_s[li * 2 + lj] * a_coeffs[block]
            
            # v: B_r is 4x4 (columns 0-3 of B), formed from B blocks
            for block in range(4):
                if b_coeffs[block] != 0:
                    br = block_row_start[block]
                    bc = block_col_start[block]
                    for lj in range(2):
                        for lk in range(2):
                            gj = br + lj
                            gk = bc + lk
                            v[gj * p + gk] += v_s[lj * 2 + lk] * b_coeffs[block]
            
            # w: C contribution. M_r contributes to C blocks with coefficients c_coeffs.
            for block in range(4):
                if c_coeffs[block] != 0:
                    br = block_row_start[block]
                    bc = block_col_start[block]
                    for li in range(2):
                        for lj in range(2):
                            gi = br + li
                            gk = bc + lj
                            w[gk * n + gi] += w_s[li * 2 + lj] * c_coeffs[block]
            
            add_term(u, v, w)
    
    # ============================================================
    # PART 2: 4x4x2 multiplication (columns k=4,5 of B)
    # Using Strassen recursion: 7 multiplications of 2x2x1
    # Each 2x2x1 uses rank 4 (schoolbook or better)
    # Actually, 2x2x1 has rank 4 (it's just 4 dot products)
    # Total: 28 terms
    # ============================================================
    
    # For 2x2x1: A is 2x2, B is 2x1, C is 2x1
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Rank 4: each term is one product
    # u = [A00, A01, A10, A11], v = [B0, B1], w = [C0, C1]
    
    strassen_2x2x1 = [
        # (u, v, w)
        ([1, 0, 0, 0], [1, 0], [1, 0]),  # A00*B0 -> C0
        ([0, 1, 0, 0], [0, 1], [1, 0]),  # A01*B1 -> C0
        ([0, 0, 1, 0], [1, 0], [0, 1]),  # A10*B0 -> C1
        ([0, 0, 0, 1], [0, 1], [0, 1]),  # A11*B1 -> C1
    ]
    
    # For 4x4x2:
    # Split A into 2x2 blocks as before
    # B (4x2) split into 2x1 blocks: B = [[B00, B01], [B10, B11]] where each is 2x1
    # C (4x2) split into 2x1 blocks: C = [[C00, C01], [C10, C11]] where each is 2x1
    #
    # Strassen for 4x4x2:
    # M1 = (A00+A11)(B00+B10)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    outer_strassen_2x1 = [
        # r, A_coeffs, B_coeffs [B00,B01,B10,B11] (each 2x1), C_coeffs
        (1, [1,0,0,1], [1,0,1,0], [1,0,0,1]),   # M1
        (2, [0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2
        (3, [1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # M3
        (4, [0,0,0,1], [0,-1,1,0], [1,0,1,0]),  # M4
        (5, [1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        (6, [0,0,1,-1], [1,1,0,0], [0,0,0,1]),  # M6
        (7, [0,1,0,-1], [0,0,1,1], [1,0,0,1]),  # M7
    ]
    
    # B blocks for 4x2:
    # B00: rows 0-1, col 4
    # B01: rows 0-1, col 5
    # B10: rows 2-3, col 4
    # B11: rows 2-3, col 5
    b2x1_row_start = [0, 0, 2, 2]
    b2x1_col = [4, 5, 4, 5]
    
    # For each outer term r and inner term s:
    for r_idx, (r, a_coeffs, b_coeffs, c_coeffs) in enumerate(outer_strassen_2x1):
        for s_idx, (u_s, v_s, w_s) in enumerate(strassen_2x2x1):
            u = zero_u()
            v = zero_v()
            w = zero_w()
            
            # u: A_r is 4x4
            for block in range(4):
                if a_coeffs[block] != 0:
                    br = block_row_start[block]
                    bc = block_col_start[block]
                    for li in range(2):
                        for lj in range(2):
                            gi = br + li
                            gj = bc + lj
                            u[gi * m + gj] += u_s[li * 2 + lj] * a_coeffs[block]
            
            # v: B_r is 4x2
            for block in range(4):
                if b_coeffs[block] != 0:
                    br = b2x1_row_start[block]
                    gk = b2x1_col[block]
                    for lj in range(2):
                        gj = br + lj
                        v[gj * p + gk] += v_s[lj] * b_coeffs[block]
            
            # w: C contribution
            for block in range(4):
                if c_coeffs[block] != 0:
                    br = block_row_start[block]
                    gk = b2x1_col[block]
                    for li in range(2):
                        gi = br + li
                        w[gk * n + gi] += w_s[li] * c_coeffs[block]
            
            add_term(u, v, w)
    
    # ============================================================
    # Verify the decomposition
    # ============================================================
    rank = len(terms)
    
    # Build U, V, W matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r_idx, (u, v, w) in enumerate(terms):
        U[:, r_idx] = u
        V[:, r_idx] = v
        W[:, r_idx] = w
    
    # Verify: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all i,j,k
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # Reshape for easier checking
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Check all entries
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        # Convert to lists
        U = U.tolist()
        V = V.tolist()
        W = W.tolist()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
