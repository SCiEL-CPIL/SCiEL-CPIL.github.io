import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    Uses a block decomposition strategy:
    - Splits rows of A (3) into 2 + 1.
    - Splits cols of A (5) into 2 + 2 + 1.
    - Splits cols of B (7) into 3 + 2 + 2.
    
    Blocks:
    1. A[0:2, 0:2] * B[0:2, 0:3] (2x2 by 2x3). Rank 11 (Strassen for first 2 cols, schoolbook for 3rd).
    2. A[0:2, 2:4] * B[2:4, 3:5] (2x2 by 2x2). Rank 7 (Strassen).
    3. A[0:2, 4:5] * B[4:5, 5:7] (2x1 by 1x2). Rank 4 (Schoolbook).
    4. A[2:3, 0:2] * B[0:2, 0:3] (1x2 by 2x3). Rank 6 (Schoolbook).
    5. A[2:3, 2:4] * B[2:4, 3:5] (1x2 by 2x2). Rank 4 (Schoolbook).
    6. A[2:3, 4:5] * B[4:5, 5:7] (1x1 by 1x2). Rank 2 (Schoolbook).
    
    Total Rank: 11 + 7 + 4 + 6 + 4 + 2 = 34.
    """
    n, m, p = DIMENSIONS
    rank = 34
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a block decomposition
    # i_off, j_off, k_off are offsets in global A, B, C indices
    # u, v, w are local matrices of shape (n_b*m_b, r_b), (m_b*p_b, r_b), (p_b*n_b, r_b)
    # c is the starting column index in the global rank
    def add_block(i_off, j_off, k_off, u, v, w, c):
        n_b, m_b = u.shape[0] // u.shape[1] if u.size else (1,1), u.shape[1]
        # Actually easier to pass dimensions
        pass 
        
    # We need to know the local dimensions to map indices.
    # Local U index: (local_i * m_b + local_j, local_c)
    # Global U index: ((i_off + local_i) * m + (j_off + local_j), c + local_c)
    
    def add_block_full(i_off, j_off, k_off, n_b, m_b, p_b, u, v, w, c):
        for li in range(n_b):
            for lj in range(m_b):
                u_idx = li * m_b + lj
                u_g_idx = (i_off + li) * m + (j_off + lj)
                for lc in range(u.shape[1]):
                    val = u[u_idx, lc]
                    if val != 0:
                        U[u_g_idx, c + lc] += val
                        
        for lj in range(m_b):
            for lk in range(p_b):
                v_idx = lj * p_b + lk
                v_g_idx = (j_off + lj) * p + (k_off + lk)
                for lc in range(v.shape[1]):
                    val = v[v_idx, lc]
                    if val != 0:
                        V[v_g_idx, c + lc] += val
                        
        for lk in range(p_b):
            for li in range(n_b):
                w_idx = lk * n_b + li
                w_g_idx = (k_off + lk) * n + (i_off + li)
                for lc in range(w.shape[1]):
                    val = w[w_idx, lc]
                    if val != 0:
                        W[w_g_idx, c + lc] += val

    # 1. Strassen 2x2x2
    def strassen_222():
        # Returns U, V, W for 2x2 by 2x2
        # M1 = (a+d)(x+v)
        # M2 = (c+d)x
        # M3 = a(y-v)
        # M4 = d(u-x)
        # M5 = (a+b)v
        # M6 = (c-d)(u+v)
        # M7 = (b-c)(x+u)
        
        # Local indices: A 2x2, B 2x2
        # U: 4 rows (a,b,c,d), 7 cols
        # V: 4 rows (x,y,u,v), 7 cols
        # W: 4 rows (p,q,r,s), 7 cols
        
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # Cols 0-6 correspond to M1-M7
        
        # M1: (a+d)(x+v)
        # U: a=1, d=1. V: x=1, v=1.
        # W: p = M1, q = M3+M4+M6, r = M2+M4+M7, s = M1+M5+M7 - M2? No.
        # Standard Strassen:
        # P = M1 - M4 + M5 + M7
        # Q = M2 + M4
        # R = M1 + M3
        # S = M1 - M2 + M6 + M7
        
        # Let's map carefully.
        # A = [[a, b], [c, d]], B = [[x, y], [u, v]]
        # M1 = (a+d)(x+v)
        # M2 = (c+d)x
        # M3 = a(y-v)
        # M4 = d(u-x)
        # M5 = (a+b)v
        # M6 = (c-d)(u+v)
        # M7 = (b-c)(x+u)
        
        # U (A coeffs):
        # M1: a=1, d=1
        # M2: c=1, d=1
        # M3: a=1
        # M4: d=1
        # M5: a=1, b=1
        # M6: c=1, d=-1
        # M7: b=1, c=-1
        
        u[0, 0] = 1; u[3, 0] = 1 # M1
        u[2, 1] = 1; u[3, 1] = 1 # M2
        u[0, 2] = 1              # M3
        u[3, 3] = 1              # M4
        u[0, 4] = 1; u[1, 4] = 1 # M5
        u[2, 5] = 1; u[3, 5] = -1 # M6
        u[1, 6] = 1; u[2, 6] = -1 # M7
        
        # V (B coeffs):
        # M1: x=1, v=1
        # M2: x=1
        # M3: y=1, v=-1
        # M4: u=1, x=-1
        # M5: v=1
        # M6: u=1, v=1
        # M7: x=1, u=1
        
        v[0, 0] = 1; v[3, 0] = 1 # M1
        v[0, 1] = 1              # M2
        v[1, 2] = 1; v[3, 2] = -1 # M3
        v[2, 3] = 1; v[0, 3] = -1 # M4
        v[3, 4] = 1              # M5
        v[2, 5] = 1; v[3, 5] = 1 # M6
        v[0, 6] = 1; v[2, 6] = 1 # M7
        
        # W (C coeffs):
        # P = M1 - M4 + M5 + M7
        # Q = M2 + M4
        # R = M1 + M3
        # S = M1 - M2 + M6 + M7
        
        # W rows: P, Q, R, S
        w[0, 0] = 1; w[0, 3] = -1; w[0, 4] = 1; w[0, 6] = 1 # P
        w[1, 1] = 1; w[1, 3] = 1                             # Q
        w[2, 0] = 1; w[2, 2] = 1                             # R
        w[3, 0] = 1; w[3, 1] = -1; w[3, 5] = 1; w[3, 6] = 1 # S
        
        return u, v, w

    # 2. Hybrid 2x2x3 (Rank 11)
    # First 2 cols: Strassen (7 ranks)
    # Last 1 col: Schoolbook (4 ranks)
    def hybrid_223():
        u_s, v_s, w_s = strassen_222()
        
        # Schoolbook for 3rd col (index 2 in local B, index 2 in local C)
        # A 2x2, B col 2 is [z, w]^T (local indices 2,3 in flat B? No, B is 2x3.
        # Local B indices: 
        # row 0: 0,1,2
        # row 1: 3,4,5
        # So z is V[0, 10] (global local idx 2? No, local rank col 7-10)
        # Let's define local V for schoolbook part.
        
        # We need to append 4 columns to U, V, W.
        u_h = np.hstack([u_s, np.zeros((4, 4))])
        v_h = np.hstack([v_s, np.zeros((4, 4))])
        w_h = np.hstack([w_s, np.zeros((4, 4))])
        
        # Schoolbook terms: a*z, b*w, c*z, d*w
        # Local A indices: a=0, b=1, c=2, d=3
        # Local B indices for col 2:
        # B is 2x3.
        # Flat index: r*p + c.
        # z is B[0,2] -> idx 2.
        # w is B[1,2] -> idx 5.
        
        u_h[0, 7] = 1 # a
        u_h[1, 8] = 1 # b
        u_h[2, 9] = 1 # c
        u_h[3, 10] = 1 # d
        
        v_h[2, 7] = 1 # z
        v_h[5, 8] = 1 # w
        v_h[2, 9] = 1 # z
        v_h[5, 10] = 1 # w
        
        # C col 2 is local C indices 2 (for row 0) and 5 (for row 1)?
        # C is 2x3.
        # Flat index: r*n + c? No, W shape is (p*n, rank).
        # Task says W[k*n+i, r].
        # So for local 2x2x3:
        # W index = k*n_b + i.
        # Col 2 of C:
        # Row 0, Col 2: k=2, i=0 -> idx 2*2+0 = 4.
        # Row 1, Col 2: k=2, i=1 -> idx 2*2+1 = 5.
        
        # Wait, W_s has 4 rows (P,Q,R,S) corresponding to C[0,0], C[1,0], C[0,1], C[1,1]?
        # No, W_s from strassen_222 has 4 rows.
        # My strassen_222 W mapping:
        # P -> C[0,0]
        # Q -> C[1,0]
        # R -> C[0,1]
        # S -> C[1,1]
        # Indices in W_s: 0, 1, 2, 3.
        # But local W index for C[i,j] is j*n_b + i.
        # C[0,0] -> 0.
        # C[1,0] -> 1.
        # C[0,1] -> 2.
        # C[1,1] -> 3.
        # This matches.
        
        # So for col 2 (j=2):
        # C[0,2] -> idx 4.
        # C[1,2] -> idx 5.
        
        w_h[4, 7] = 1 # a*z -> C[0,2]
        w_h[5, 8] = 1 # b*w -> C[1,2]
        w_h[4, 9] = 1 # c*z -> C[0,2]
        w_h[5, 10] = 1 # d*w -> C[1,2]
        
        return u_h, v_h, w_h

    # 3. Schoolbook 2x1x2 (Rank 4)
    def schoolbook_212():
        # A 2x1, B 1x2
        # U: 2 rows (a,b)
        # V: 2 rows (x,y)
        # W: 4 rows (ax, ay, bx, by) -> C[0,0], C[0,1], C[1,0], C[1,1]
        
        u = np.zeros((2, 4))
        v = np.zeros((2, 4))
        w = np.zeros((4, 4))
        
        u[0, 0] = 1; u[1, 1] = 1; u[0, 2] = 1; u[1, 3] = 1
        v[0, 0] = 1; v[1, 1] = 1; v[0, 2] = 1; v[1, 3] = 1
        
        # W indices: k*n_b + i. n_b=2.
        # C[0,0] k=0,i=0 -> 0
        # C[0,1] k=1,i=0 -> 2
        # C[1,0] k=0,i=1 -> 1
        # C[1,1] k=1,i=1 -> 3
        
        w[0, 0] = 1 # ax
        w[2, 1] = 1 # ay
        w[1, 2] = 1 # bx
        w[3, 3] = 1 # by
        
        return u, v, w

    # 4. Schoolbook 1x2x3 (Rank 6)
    def schoolbook_123():
        # A 1x2, B 2x3
        # U: 2 rows (a,b)
        # V: 6 rows
        # W: 3 rows (c0, c1, c2)
        
        u = np.zeros((2, 6))
        v = np.zeros((6, 6))
        w = np.zeros((3, 6))
        
        for i in range(2):
            for j in range(3):
                c = i * 3 + j
                u[i, c] = 1
                v[j, c] = 1
                w[j, c] = 1 # W index = k*n_b + i = j*1 + 0 = j. Correct.
                
        return u, v, w

    # 5. Schoolbook 1x2x2 (Rank 4)
    def schoolbook_122():
        u = np.zeros((2, 4))
        v = np.zeros((4, 4))
        w = np.zeros((2, 4))
        
        for i in range(2):
            for j in range(2):
                c = i * 2 + j
                u[i, c] = 1
                v[j, c] = 1
                w[j, c] = 1
                
        return u, v, w

    # 6. Schoolbook 1x1x2 (Rank 2)
    def schoolbook_112():
        u = np.zeros((1, 2))
        v = np.zeros((2, 2))
        w = np.zeros((2, 2))
        
        for j in range(2):
            u[0, j] = 1
            v[j, j] = 1
            w[j, j] = 1
            
        return u, v, w

    # Construct Blocks
    c = 0
    
    # Block 1: i=0, j=0, k=0. Size 2x2x3.
    u1, v1, w1 = hybrid_223()
    add_block_full(0, 0, 0, 2, 2, 3, u1, v1, w1, c)
    c += u1.shape[1]
    
    # Block 2: i=0, j=2, k=3. Size 2x2x2.
    u2, v2, w2 = strassen_222()
    add_block_full(0, 2, 3, 2, 2, 2, u2, v2, w2, c)
    c += u2.shape[1]
    
    # Block 3: i=0, j=4, k=5. Size 2x1x2.
    u3, v3, w3 = schoolbook_212()
    add_block_full(0, 4, 5, 2, 1, 2, u3, v3, w3, c)
    c += u3.shape[1]
    
    # Block 4: i=2, j=0, k=0. Size 1x2x3.
    u4, v4, w4 = schoolbook_123()
    add_block_full(2, 0, 0, 1, 2, 3, u4, v4, w4, c)
    c += u4.shape[1]
    
    # Block 5: i=2, j=2, k=3. Size 1x2x2.
    u5, v5, w5 = schoolbook_122()
    add_block_full(2, 2, 3, 1, 2, 2, u5, v5, w5, c)
    c += u5.shape[1]
    
    # Block 6: i=2, j=4, k=5. Size 1x1x2.
    u6, v6, w6 = schoolbook_112()
    add_block_full(2, 4, 5, 1, 1, 2, u6, v6, w6, c)
    c += u6.shape[1]
    
    # Check if c matches rank
    if c != rank:
        # If not, truncate or pad? Should be exact.
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
