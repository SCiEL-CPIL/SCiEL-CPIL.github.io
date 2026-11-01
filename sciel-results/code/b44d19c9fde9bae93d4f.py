"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We build the matrices as lists of lists or numpy arrays.
    # Let's use numpy for easier manipulation, then convert to lists if needed.
    # The interface expects lists of lists or arrays? "factor matrices of shapes... with integer/half-integer entries"
    # Usually lists of lists are safe.
    
    # We will accumulate the columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_column(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def get_u_idx(i, j):
        return i * m + j
        
    def get_v_idx(j, k):
        return j * p + k
        
    def get_w_idx(k, i):
        return k * n + i

    # Initialize vectors with zeros
    zero_u = [0] * (n * m)
    zero_v = [0] * (m * p)
    zero_w = [0] * (p * n)
    
    def make_u(indices_values):
        vec = [0] * (n * m)
        for idx, val in indices_values:
            vec[idx] = val
        return vec
        
    def make_v(indices_values):
        vec = [0] * (m * p)
        for idx, val in indices_values:
            vec[idx] = val
        return vec
        
    def make_w(indices_values):
        vec = [0] * (p * n)
        for idx, val in indices_values:
            vec[idx] = val
        return vec

    # Strassen blocks for p in {0,1}, {2,3}, {4,5}, {6,7}
    for k_start in range(0, 8, 2):
        k0 = k_start
        k1 = k_start + 1
        
        # Strassen products P1..P7
        # A2 is rows 0,1 and cols 1,2 of A
        # B2 block is cols k0, k1 of rows 1,2 of B
        
        # P1 = (A01 + A12) * (B1k0 + B1k1 + B2k0 + B2k1) -> Wait, Strassen is (a+d)(e+h)
        # a=A01, d=A12, e=B1k0, h=B2k1? No.
        # Matrix A2 = [[A01, A02], [A11, A12]]
        # Matrix B2_block = [[B1k0, B1k1], [B2k0, B2k1]]
        # a=A01, b=A02, c=A11, d=A12
        # e=B1k0, f=B1k1, g=B2k0, h=B2k1
        
        # P1 = (a+d)(e+h)
        u1 = make_u([(get_u_idx(0,1), 1), (get_u_idx(1,2), 1)])
        v1 = make_v([(get_v_idx(1,k0), 1), (get_v_idx(2,k1), 1)])
        # W coeffs: C0k0:1, C0k1:0, C1k0:0, C1k1:1
        w1 = make_w([(get_w_idx(k0,0), 1), (get_w_idx(k1,1), 1)])
        add_column(u1, v1, w1)
        
        # P2 = (c+d)e
        u2 = make_u([(get_u_idx(1,1), 1), (get_u_idx(1,2), 1)])
        v2 = make_v([(get_v_idx(1,k0), 1)])
        # W coeffs: C0k0:0, C0k1:0, C1k0:1, C1k1:-1
        w2 = make_w([(get_w_idx(k0,1), 1), (get_w_idx(k1,1), -1)])
        add_column(u2, v2, w2)
        
        # P3 = a(f-h)
        u3 = make_u([(get_u_idx(0,1), 1)])
        v3 = make_v([(get_v_idx(1,k1), 1), (get_v_idx(2,k1), -1)])
        # W coeffs: C0k0:0, C0k1:1, C1k0:0, C1k1:1
        w3 = make_w([(get_w_idx(k1,0), 1), (get_w_idx(k1,1), 1)])
        add_column(u3, v3, w3)
        
        # P4 = d(g-e)
        u4 = make_u([(get_u_idx(1,2), 1)])
        v4 = make_v([(get_v_idx(2,k0), 1), (get_v_idx(1,k0), -1)])
        # W coeffs: C0k0:1, C0k1:0, C1k0:1, C1k1:0
        w4 = make_w([(get_w_idx(k0,0), 1), (get_w_idx(k0,1), 1)])
        add_column(u4, v4, w4)
        
        # P5 = (a+b)h
        u5 = make_u([(get_u_idx(0,1), 1), (get_u_idx(0,2), 1)])
        v5 = make_v([(get_v_idx(2,k1), 1)])
        # W coeffs: C0k0:-1, C0k1:1, C1k0:0, C1k1:0
        w5 = make_w([(get_w_idx(k0,0), -1), (get_w_idx(k1,0), 1)])
        add_column(u5, v5, w5)
        
        # P6 = (c-a)(e+f)
        u6 = make_u([(get_u_idx(1,1), 1), (get_u_idx(0,1), -1)])
        v6 = make_v([(get_v_idx(1,k0), 1), (get_v_idx(1,k1), 1)])
        # W coeffs: C0k0:0, C0k1:0, C1k0:0, C1k1:1
        w6 = make_w([(get_w_idx(k1,1), 1)])
        add_column(u6, v6, w6)
        
        # P7 = (b-d)(g+h)
        u7 = make_u([(get_u_idx(0,2), 1), (get_u_idx(1,2), -1)])
        v7 = make_v([(get_v_idx(2,k0), 1), (get_v_idx(2,k1), 1)])
        # W coeffs: C0k0:1, C0k1:0, C1k0:0, C1k1:0
        w7 = make_w([(get_w_idx(k0,0), 1)])
        add_column(u7, v7, w7)
        
        # Schoolbook for A1 * B1_block (2x1 * 1x2)
        # A1 is col 0 of A. B1_block is row 0 of B, cols k0, k1.
        # Q1 = A00 * B0k0
        uq1 = make_u([(get_u_idx(0,0), 1)])
        vq1 = make_v([(get_v_idx(0,k0), 1)])
        wq1 = make_w([(get_w_idx(k0,0), 1)])
        add_column(uq1, vq1, wq1)
        
        # Q2 = A00 * B0k1
        uq2 = make_u([(get_u_idx(0,0), 1)])
        vq2 = make_v([(get_v_idx(0,k1), 1)])
        wq2 = make_w([(get_w_idx(k1,0), 1)])
        add_column(uq2, vq2, wq2)
        
        # Q3 = A10 * B0k0
        uq3 = make_u([(get_u_idx(1,0), 1)])
        vq3 = make_v([(get_v_idx(0,k0), 1)])
        wq3 = make_w([(get_w_idx(k0,1), 1)])
        add_column(uq3, vq3, wq3)
        
        # Q4 = A10 * B0k1
        uq4 = make_u([(get_u_idx(1,0), 1)])
        vq4 = make_v([(get_v_idx(0,k1), 1)])
        wq4 = make_w([(get_w_idx(k1,1), 1)])
        add_column(uq4, vq4, wq4)

    # Last block k=8
    k8 = 8
    
    # Schoolbook for A2 * B2_last (2x2 * 2x1)
    # R1 = A01 * B1k8
    ur1 = make_u([(get_u_idx(0,1), 1)])
    vr1 = make_v([(get_v_idx(1,k8), 1)])
    wr1 = make_w([(get_w_idx(k8,0), 1)])
    add_column(ur1, vr1, wr1)
    
    # R2 = A02 * B1k8
    ur2 = make_u([(get_u_idx(0,2), 1)])
    vr2 = make_v([(get_v_idx(1,k8), 1)])
    wr2 = make_w([(get_w_idx(k8,0), 1)])
    add_column(ur2, vr2, wr2)
    
    # R3 = A11 * B1k8
    ur3 = make_u([(get_u_idx(1,1), 1)])
    vr3 = make_v([(get_v_idx(1,k8), 1)])
    wr3 = make_w([(get_w_idx(k8,1), 1)])
    add_column(ur3, vr3, wr3)
    
    # R4 = A12 * B1k8
    ur4 = make_u([(get_u_idx(1,2), 1)])
    vr4 = make_v([(get_v_idx(1,k8), 1)])
    wr4 = make_w([(get_w_idx(k8,1), 1)])
    add_column(ur4, vr4, wr4)
    
    # Schoolbook for A1 * B1_last (2x1 * 1x1)
    # S1 = A00 * B0k8
    us1 = make_u([(get_u_idx(0,0), 1)])
    vs1 = make_v([(get_v_idx(0,k8), 1)])
    ws1 = make_w([(get_w_idx(k8,0), 1)])
    add_column(us1, vs1, ws1)
    
    # S2 = A10 * B0k8
    us2 = make_u([(get_u_idx(1,0), 1)])
    vs2 = make_v([(get_v_idx(0,k8), 1)])
    ws2 = make_w([(get_w_idx(k8,1), 1)])
    add_column(us2, vs2, ws2)

    # Convert to numpy arrays and then to lists if necessary
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
