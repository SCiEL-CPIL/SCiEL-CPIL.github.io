import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W lists of lists.
    # U: (n*m, rank) -> (15, rank)
    # V: (m*p, rank) -> (30, rank)
    # W: (p*n, rank) -> (18, rank)
    
    # We don't know the rank in advance, so we'll use lists and append.
    U_rows = [[0] * 15 for _ in range(0)] # Placeholder, we will build columns
    # Better: Build columns as we go.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_factor(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Helper to create zero vectors
    def zero_u(): return [0.0] * 15
    def zero_v(): return [0.0] * 30
    def zero_w(): return [0.0] * 18
    
    # Indices:
    # A[i, j] -> U index i*m + j
    # B[j, k] -> V index j*p + k
    # C[i, k] -> W index k*n + i (Trace convention: W[k*n+i, r])
    
    # --- Part 1: A1 B1 (3x2 * 2x6) ---
    # A1 is A[0:3, 0:2]
    # B1 is B[0:2, 0:6]
    
    # A1a: A[0:2, 0:2]
    # A1b: A[2:3, 0:2]
    
    # B1a: B[0:2, 0:2]
    # B1b: B[0:2, 2:4]
    # B1c: B[0:2, 4:6]
    
    # 1. A1a B1a (2x2 * 2x2) -> Strassen
    # Strassen components for X=A1a, Y=B1a
    # X indices: i in 0,1; j in 0,1. U idx: i*5+j
    # Y indices: j in 0,1; k in 0,1. V idx: j*6+k
    # Z indices: i in 0,1; k in 0,1. W idx: k*3+i
    
    # M1 = (X00+X11)(Y00+Y11)
    u1 = zero_u(); u1[0] = 1; u1[11] = 1
    v1 = zero_v(); v1[0] = 1; v1[7] = 1
    w1 = zero_w()
    # Z00 += M1, Z10 += M1
    w1[0] = 1; w1[3] = 1 # k=0,i=0 -> 0; k=0,i=1 -> 3
    add_factor(u1, v1, w1)
    
    # M2 = (X10+X11)Y00
    u2 = zero_u(); u2[5] = 1; u2[11] = 1
    v2 = zero_v(); v2[0] = 1
    w2 = zero_w()
    # Z01 += M2, Z11 -= M2
    w2[1] = 1; w2[4] = -1 # k=1,i=0 -> 1; k=1,i=1 -> 4
    add_factor(u2, v2, w2)
    
    # M3 = X00(Y10-Y11)
    u3 = zero_u(); u3[0] = 1
    v3 = zero_v(); v3[6] = 1; v3[7] = -1
    w3 = zero_w()
    # Z01 += M3, Z10 -= M3
    w3[1] = 1; w3[3] = -1
    add_factor(u3, v3, w3)
    
    # M4 = X11(Y01-Y00)
    u4 = zero_u(); u4[11] = 1
    v4 = zero_v(); v4[1] = 1; v4[0] = -1
    w4 = zero_w()
    # Z00 += M4, Z01 -= M4, Z11 += M4
    w4[0] = 1; w4[1] = -1; w4[4] = 1
    add_factor(u4, v4, w4)
    
    # M5 = (X00+X10)Y11
    u5 = zero_u(); u5[0] = 1; u5[5] = 1
    v5 = zero_v(); v5[7] = 1
    w5 = zero_w()
    # Z00 -= M5, Z10 += M5
    w5[0] = -1; w5[3] = 1
    add_factor(u5, v5, w5)
    
    # M6 = (X01-X00)(Y00+Y10)
    u6 = zero_u(); u6[1] = 1; u6[0] = -1
    v6 = zero_v(); v6[0] = 1; v6[6] = 1
    w6 = zero_w()
    # Z01 += M6, Z11 -= M6
    w6[1] = 1; w6[4] = -1
    add_factor(u6, v6, w6)
    
    # M7 = (X10-X11)(Y01+Y11)
    u7 = zero_u(); u7[5] = 1; u7[11] = -1
    v7 = zero_v(); v7[1] = 1; v7[7] = 1
    w7 = zero_w()
    # Z00 += M7, Z10 += M7
    w7[0] = 1; w7[3] = 1
    add_factor(u7, v7, w7)
    
    # 2. A1a B1b (2x2 * 2x2) -> Strassen
    # Y indices: j in 0,1; k in 2,3. V idx: j*6+k
    # Z indices: i in 0,1; k in 2,3. W idx: k*3+i
    
    # M1
    u = zero_u(); u[0] = 1; u[11] = 1
    v = zero_v(); v[2] = 1; v[9] = 1
    w = zero_w()
    w[2] = 1; w[5] = 1
    add_factor(u, v, w)
    
    # M2
    u = zero_u(); u[5] = 1; u[11] = 1
    v = zero_v(); v[2] = 1
    w = zero_w()
    w[3] = 1; w[6] = -1
    add_factor(u, v, w)
    
    # M3
    u = zero_u(); u[0] = 1
    v = zero_v(); v[8] = 1; v[9] = -1
    w = zero_w()
    w[3] = 1; w[5] = -1
    add_factor(u, v, w)
    
    # M4
    u = zero_u(); u[11] = 1
    v = zero_v(); v[3] = 1; v[2] = -1
    w = zero_w()
    w[2] = 1; w[3] = -1; w[6] = 1
    add_factor(u, v, w)
    
    # M5
    u = zero_u(); u[0] = 1; u[5] = 1
    v = zero_v(); v[9] = 1
    w = zero_w()
    w[2] = -1; w[5] = 1
    add_factor(u, v, w)
    
    # M6
    u = zero_u(); u[1] = 1; u[0] = -1
    v = zero_v(); v[2] = 1; v[8] = 1
    w = zero_w()
    w[3] = 1; w[6] = -1
    add_factor(u, v, w)
    
    # M7
    u = zero_u(); u[5] = 1; u[11] = -1
    v = zero_v(); v[3] = 1; v[9] = 1
    w = zero_w()
    w[2] = 1; w[5] = 1
    add_factor(u, v, w)
    
    # 3. A1a B1c (2x2 * 2x2) -> Strassen
    # Y indices: j in 0,1; k in 4,5. V idx: j*6+k
    # Z indices: i in 0,1; k in 4,5. W idx: k*3+i
    
    # M1
    u = zero_u(); u[0] = 1; u[11] = 1
    v = zero_v(); v[4] = 1; v[11] = 1
    w = zero_w()
    w[4] = 1; w[7] = 1
    add_factor(u, v, w)
    
    # M2
    u = zero_u(); u[5] = 1; u[11] = 1
    v = zero_v(); v[4] = 1
    w = zero_w()
    w[5] = 1; w[8] = -1
    add_factor(u, v, w)
    
    # M3
    u = zero_u(); u[0] = 1
    v = zero_v(); v[10] = 1; v[11] = -1
    w = zero_w()
    w[5] = 1; w[7] = -1
    add_factor(u, v, w)
    
    # M4
    u = zero_u(); u[11] = 1
    v = zero_v(); v[5] = 1; v[4] = -1
    w = zero_w()
    w[4] = 1; w[5] = -1; w[8] = 1
    add_factor(u, v, w)
    
    # M5
    u = zero_u(); u[0] = 1; u[5] = 1
    v = zero_v(); v[11] = 1
    w = zero_w()
    w[4] = -1; w[7] = 1
    add_factor(u, v, w)
    
    # M6
    u = zero_u(); u[1] = 1; u[0] = -1
    v = zero_v(); v[4] = 1; v[10] = 1
    w = zero_w()
    w[5] = 1; w[8] = -1
    add_factor(u, v, w)
    
    # M7
    u = zero_u(); u[5] = 1; u[11] = -1
    v = zero_v(); v[5] = 1; v[11] = 1
    w = zero_w()
    w[4] = 1; w[7] = 1
    add_factor(u, v, w)
    
    # 4. A1b B1a (1x2 * 2x2) -> Schoolbook
    # A1b: i=2, j in 0,1. U idx: 2*5+j = 10, 11
    # B1a: j in 0,1, k in 0,1. V idx: j*6+k
    # Z: i=2, k in 0,1. W idx: k*3+2
    
    # X00 Y00
    u = zero_u(); u[10] = 1
    v = zero_v(); v[0] = 1
    w = zero_w(); w[2] = 1
    add_factor(u, v, w)
    
    # X00 Y01
    u = zero_u(); u[10] = 1
    v = zero_v(); v[1] = 1
    w = zero_w(); w[5] = 1
    add_factor(u, v, w)
    
    # X01 Y00
    u = zero_u(); u[11] = 1
    v = zero_v(); v[6] = 1
    w = zero_w(); w[2] = 1
    add_factor(u, v, w)
    
    # X01 Y01
    u = zero_u(); u[11] = 1
    v = zero_v(); v[7] = 1
    w = zero_w(); w[5] = 1
    add_factor(u, v, w)
    
    # 5. A1b B1b (1x2 * 2x2) -> Schoolbook
    # B1b: k in 2,3. V idx: j*6+k
    # Z: i=2, k in 2,3. W idx: k*3+2
    
    u = zero_u(); u[10] = 1
    v = zero_v(); v[2] = 1
    w = zero_w(); w[8] = 1
    add_factor(u, v, w)
    
    u = zero_u(); u[10] = 1
    v = zero_v(); v[3] = 1
    w = zero_w(); w[11] = 1
    add_factor(u, v, w)
    
    u = zero_u(); u[11] = 1
    v = zero_v(); v[8] = 1
    w = zero_w(); w[8] = 1
    add_factor(u, v, w)
    
    u = zero_u(); u[11] = 1
    v = zero_v(); v[9] = 1
    w = zero_w(); w[11] = 1
    add_factor(u, v, w)
    
    # 6. A1b B1c (1x2 * 2x2) -> Schoolbook
    # B1c: k in 4,5. V idx: j*6+k
    # Z: i=2, k in 4,5. W idx: k*3+2
    
    u = zero_u(); u[10] = 1
    v = zero_v(); v[4] = 1
    w = zero_w(); w[14] = 1
    add_factor(u, v, w)
    
    u = zero_u(); u[10] = 1
    v = zero_v(); v[5] = 1
    w = zero_w(); w[17] = 1
    add_factor(u, v, w)
    
    u = zero_u(); u[11] = 1
    v = zero_v(); v[10] = 1
    w = zero_w(); w[14] = 1
    add_factor(u, v, w)
    
    u = zero_u(); u[11] = 1
    v = zero_v(); v[11] = 1
    w = zero_w(); w[17] = 1
    add_factor(u, v, w)
    
    # --- Part 2: A2 B2 (3x3 * 3x6) ---
    # A2: i in 0,1,2; j in 0,1,2. U idx: i*5+j
    # B2: j in 0,1,2; k in 0..5. V idx: j*6+k
    # Z: i in 0,1,2; k in 0..5. W idx: k*3+i
    
    # B2a: k in 0,1
    for j in range(3):
        for i in range(3):
            for k in range(2):
                u = zero_u(); u[i*5+j] = 1
                v = zero_v(); v[j*6+k] = 1
                w = zero_w(); w[k*3+i] = 1
                add_factor(u, v, w)
                
    # B2b: k in 2,3
    for j in range(3):
        for i in range(3):
            for k in range(2, 4):
                u = zero_u(); u[i*5+j] = 1
                v = zero_v(); v[j*6+k] = 1
                w = zero_w(); w[k*3+i] = 1
                add_factor(u, v, w)
                
    # B2c: k in 4,5
    for j in range(3):
        for i in range(3):
            for k in range(4, 6):
                u = zero_u(); u[i*5+j] = 1
                v = zero_v(); v[j*6+k] = 1
                w = zero_w(); w[k*3+i] = 1
                add_factor(u, v, w)
                
    # Convert to required format
    rank = len(U_cols)
    U = [[0] * rank for _ in range(15)]
    V = [[0] * rank for _ in range(30)]
    W = [[0] * rank for _ in range(18)]
    
    for r in range(rank):
        for i in range(15):
            U[i][r] = U_cols[r][i]
        for i in range(30):
            V[i][r] = V_cols[r][i]
        for i in range(18):
            W[i][r] = W_cols[r][i]
            
    return {"U": U, "V": V, "W": W}
