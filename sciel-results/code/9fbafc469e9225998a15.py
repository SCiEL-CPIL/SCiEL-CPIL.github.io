import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Shapes
    dim_u = n * m  # 12
    dim_v = m * p  # 32
    dim_w = p * n  # 24
    
    # We will build the decomposition with rank 88
    target_rank = 88
    U = np.zeros((dim_u, target_rank))
    V = np.zeros((dim_v, target_rank))
    W = np.zeros((dim_w, target_rank))
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        if col >= target_rank:
            return
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec
        col += 1

    def get_u(i, j):
        v = np.zeros(dim_u)
        v[i * m + j] = 1.0
        return v

    def get_v(j, k):
        v = np.zeros(dim_v)
        v[j * p + k] = 1.0
        return v

    def get_w(k, i):
        v = np.zeros(dim_w)
        v[k * n + i] = 1.0
        return v

    # Part 1: C_2 = A_2 B (1x4 * 4x8)
    # A_2 is row 2 of A (i=2)
    # 32 terms
    for j in range(m):
        for k in range(p):
            u = get_u(2, j)
            v = get_v(j, k)
            w = get_w(k, 2)
            add_term(u, v, w)
            
    # Part 2: C_1 = A_1 B (2x4 * 4x8)
    # Split A_1 (rows 0,1) into A_1L (cols 0,1) and A_1R (cols 2,3)
    # Split B (4x8) into B_L (rows 0,1) and B_R (rows 2,3)
    # C_1 = A_1L B_L + A_1R B_R
    
    # Helper for Strassen M_{2,2,2}
    # a00, a01, a10, a11 are indices in A block
    # b00, b01, b10, b11 are indices in B block
    # c00, c01, c10, c11 are indices in C block
    def add_strassen(a_i0, a_j0, b_i0, b_j0, c_i0, c_j0):
        # a_i0, a_j0: base indices for A rows/cols
        # b_i0, b_j0: base indices for B rows/cols
        # c_i0, c_j0: base indices for C rows/cols
        # A block: A[a_i0:a_i0+2, a_j0:a_j0+2]
        # B block: B[b_i0:b_i0+2, b_j0:b_j0+2]
        # C block: C[c_i0:c_i0+2, c_j0:c_j0+2]
        
        # Term 1: (A00+A11)(B00+B11)
        u = get_u(a_i0, a_j0) + get_u(a_i0+1, a_j0+1)
        v = get_v(b_i0, b_j0) + get_v(b_i0+1, b_j0+1)
        w = get_w(c_j0, c_i0) - get_w(c_j0+1, c_i0+1) + get_w(c_j0+1, c_i0) - get_w(c_j0, c_i0+1)
        # Wait, W index is k*n + i.
        # C[c_i, c_j] corresponds to W[c_j * n + c_i]
        # C00 -> c_i0, c_j0 -> W[c_j0 * n + c_i0]
        # C11 -> c_i0+1, c_j0+1 -> W[(c_j0+1) * n + (c_i0+1)]
        # C10 -> c_i0+1, c_j0 -> W[c_j0 * n + (c_i0+1)]
        # C01 -> c_i0, c_j0+1 -> W[(c_j0+1) * n + c_i0]
        
        # Formula: C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M4 + M7
        # C10 = M2 + M5 - M1 + M6
        # C11 = M1 - M2 + M3 - M6
        
        # Let's construct W vectors for each M carefully.
        # M1: C00:+1, C01:0, C10:-1, C11:+1
        w1 = np.zeros(dim_w)
        w1[c_j0 * n + c_i0] = 1
        w1[c_j0 * n + (c_i0+1)] = -1
        w1[(c_j0+1) * n + (c_i0+1)] = 1
        add_term(u, v, w1)

        # M2: (A10+A11) B00
        u = get_u(a_i0+1, a_j0) + get_u(a_i0+1, a_j0+1)
        v = get_v(b_i0, b_j0)
        w2 = np.zeros(dim_w)
        w2[c_j0 * n + (c_i0+1)] = 1
        w2[(c_j0+1) * n + (c_i0+1)] = -1
        add_term(u, v, w2)

        # M3: A00 (B01-B11)
        u = get_u(a_i0, a_j0)
        v = get_v(b_i0, b_j0+1) - get_v(b_i0+1, b_j0+1)
        w3 = np.zeros(dim_w)
        w3[(c_j0+1) * n + c_i0] = 1
        w3[(c_j0+1) * n + (c_i0+1)] = 1
        add_term(u, v, w3)

        # M4: A11 (B10-B01)
        u = get_u(a_i0+1, a_j0+1)
        v = get_v(b_i0+1, b_j0) - get_v(b_i0, b_j0+1)
        w4 = np.zeros(dim_w)
        w4[c_j0 * n + c_i0] = -1
        w4[(c_j0+1) * n + c_i0] = 1
        add_term(u, v, w4)

        # M5: (A00+A01) B11
        u = get_u(a_i0, a_j0) + get_u(a_i0, a_j0+1)
        v = get_v(b_i0+1, b_j0+1)
        w5 = np.zeros(dim_w)
        w5[c_j0 * n + c_i0] = 1
        w5[c_j0 * n + (c_i0+1)] = 1
        add_term(u, v, w5)

        # M6: (A10-A00) (B00+B01)
        u = get_u(a_i0+1, a_j0) - get_u(a_i0, a_j0)
        v = get_v(b_i0, b_j0) + get_v(b_i0, b_j0+1)
        w6 = np.zeros(dim_w)
        w6[c_j0 * n + (c_i0+1)] = 1
        w6[(c_j0+1) * n + (c_i0+1)] = -1
        add_term(u, v, w6)

        # M7: (A01-A11) (B10+B11)
        u = get_u(a_i0, a_j0+1) - get_u(a_i0+1, a_j0+1)
        v = get_v(b_i0+1, b_j0) + get_v(b_i0+1, b_j0+1)
        w7 = np.zeros(dim_w)
        w7[c_j0 * n + c_i0] = -1
        w7[(c_j0+1) * n + c_i0] = 1
        add_term(u, v, w7)

    # A_1L: rows 0,1; cols 0,1. a_i0=0, a_j0=0
    # B_L: rows 0,1; cols 0..7.
    # Split B_L into 4 blocks of 2x2.
    # Block 0: cols 0,1. b_i0=0, b_j0=0.
    # Block 1: cols 2,3. b_i0=0, b_j0=2.
    # Block 2: cols 4,5. b_i0=0, b_j0=4.
    # Block 3: cols 6,7. b_i0=0, b_j0=6.
    
    # C_1L: rows 0,1; cols 0..7.
    # Block 0: cols 0,1. c_i0=0, c_j0=0.
    # Block 1: cols 2,3. c_i0=0, c_j0=2.
    # Block 2: cols 4,5. c_i0=0, c_j0=4.
    # Block 3: cols 6,7. c_i0=0, c_j0=6.
    
    for b in range(4):
        b_j0 = b * 2
        c_j0 = b * 2
        add_strassen(0, 0, 0, b_j0, 0, c_j0)
        
    # A_1R: rows 0,1; cols 2,3. a_i0=0, a_j0=2
    # B_R: rows 2,3; cols 0..7.
    # Split B_R into 4 blocks of 2x2.
    # Block 0: cols 0,1. b_i0=2, b_j0=0.
    # Block 1: cols 2,3. b_i0=2, b_j0=2.
    # Block 2: cols 4,5. b_i0=2, b_j0=4.
    # Block 3: cols 6,7. b_i0=2, b_j0=6.
    
    # C_1R: rows 0,1; cols 0..7.
    # Block 0: cols 0,1. c_i0=0, c_j0=0.
    # ...
    
    for b in range(4):
        b_j0 = b * 2
        c_j0 = b * 2
        add_strassen(0, 2, 2, b_j0, 0, c_j0)

    # Verify
    def verify(U, V, W, tol=1e-8):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U = np.zeros((n*m, rank_sb))
        V = np.zeros((m*p, rank_sb))
        W = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
