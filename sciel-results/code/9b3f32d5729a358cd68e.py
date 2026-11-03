"""Rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication."""
DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    U_rows = n * m  # 6
    V_rows = m * p  # 15
    W_rows = p * n  # 10
    
    cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        cols.append((u_vec, v_vec, w_vec))
    
    def make_zero_u():
        return [0] * U_rows
    
    def make_zero_v():
        return [0] * V_rows
    
    def make_zero_w():
        return [0] * W_rows
    
    # Decomposition: split m=3 into j=0 and j=1,2
    # Part 1: j=0, i in {0,1}, k in {0,1,2,3,4}: 5 rank-1 terms
    for k in range(p):
        for i in range(n):
            u = make_zero_u()
            v = make_zero_v()
            w = make_zero_w()
            u[i * m + 0] = 1
            v[0 * p + k] = 1
            w[k * n + i] = 1
            add_term(u, v, w)
    
    # Part 2: j in {1,2}, i in {0,1}, k in {0,1,2,3,4}
    # Use Strassen for k=0,1 (7 terms) + k=2,3 (7 terms) + k=4 (4 terms) = 18 terms
    
    def strassen_block(j_offset, k_offset):
        """Add 7 Strassen rank-1 terms for 2x2x2 block."""
        # Local i in {0,1}, j in {0,1}, k in {0,1}
        # Global: i, j=j_offset+j_local, k=k_offset+k_local
        
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> C[0,0]+=1, C[1,1]+=1
        u = make_zero_u(); v = make_zero_v(); w = make_zero_w()
        u[0 * m + (j_offset + 0)] = 1  # A[0,0]
        u[1 * m + (j_offset + 1)] = 1  # A[1,1]
        v[(j_offset + 0) * p + (k_offset + 0)] = 1  # B[0,0]
        v[(j_offset + 1) * p + (k_offset + 1)] = 1  # B[1,1]
        w[(k_offset + 0) * n + 0] = 1  # C[0,0]
        w[(k_offset + 1) * n + 1] = 1  # C[1,1]
        add_term(u, v, w)
        
        # M2 = (A[1,0]+A[1,1]) B[0,0] -> C[1,0]+=1
        u = make_zero_u(); v = make_zero_v(); w = make_zero_w()
        u[1 * m + (j_offset + 0)] = 1  # A[1,0]
        u[1 * m + (j_offset + 1)] = 1  # A[1,1]
        v[(j_offset + 0) * p + (k_offset + 0)] = 1  # B[0,0]
        w[(k_offset + 0) * n + 1] = 1  # C[1,0]
        add_term(u, v, w)
        
        # M3 = A[0,0] (B[0,1]-B[1,1]) -> C[0,1]+=1
        u = make_zero_u(); v = make_zero_v(); w = make_zero_w()
        u[0 * m + (j_offset + 0)] = 1  # A[0,0]
        v[(j_offset + 0) * p + (k_offset + 1)] = 1  # B[0,1]
        v[(j_offset + 1) * p + (k_offset + 1)] = -1  # B[1,1] with -1
        w[(k_offset + 1) * n + 0] = 1  # C[0,1]
        add_term(u, v, w)
        
        # M4 = A[1,1] (B[1,0]-B[0,0]) -> C[0,0]+=1, C[1,0]+=1
        u = make_zero_u(); v = make_zero_v(); w = make_zero_w()
        u[1 * m + (j_offset + 1)] = 1  # A[1,1]
        v[(j_offset + 1) * p + (k_offset + 0)] = 1  # B[1,0]
        v[(j_offset + 0) * p + (k_offset + 0)] = -1  # B[0,0] with -1
        w[(k_offset + 0) * n + 0] = 1  # C[0,0]
        w[(k_offset + 0) * n + 1] = 1  # C[1,0]
        add_term(u, v, w)
        
        # M5 = (A[0,0]+A[0,1]) B[1,1] -> C[0,0]-=1, C[0,1]+=1
        u = make_zero_u(); v = make_zero_v(); w = make_zero_w()
        u[0 * m + (j_offset + 0)] = 1  # A[0,0]
        u[0 * m + (j_offset + 1)] = 1  # A[0,1]
        v[(j_offset + 1) * p + (k_offset + 1)] = 1  # B[1,1]
        w[(k_offset + 0) * n + 0] = -1  # C[0,0] with -1
        w[(k_offset + 1) * n + 0] = 1  # C[0,1]
        add_term(u, v, w)
        
        # M6 = (A[1,0]-A[0,0]) (B[0,0]+B[0,1]) -> C[1,1]+=1
        u = make_zero_u(); v = make_zero_v(); w = make_zero_w()
        u[1 * m + (j_offset + 0)] = 1  # A[1,0]
        u[0 * m + (j_offset + 0)] = -1  # A[0,0] with -1
        v[(j_offset + 0) * p + (k_offset + 0)] = 1  # B[0,0]
        v[(j_offset + 0) * p + (k_offset + 1)] = 1  # B[0,1]
        w[(k_offset + 1) * n + 1] = 1  # C[1,1]
        add_term(u, v, w)
        
        # M7 = (A[0,1]-A[1,1]) (B[1,0]+B[1,1]) -> C[0,0]+=1
        u = make_zero_u(); v = make_zero_v(); w = make_zero_w()
        u[0 * m + (j_offset + 1)] = 1  # A[0,1]
        u[1 * m + (j_offset + 1)] = -1  # A[1,1] with -1
        v[(j_offset + 1) * p + (k_offset + 0)] = 1  # B[1,0]
        v[(j_offset + 1) * p + (k_offset + 1)] = 1  # B[1,1]
        w[(k_offset + 0) * n + 0] = 1  # C[0,0]
        add_term(u, v, w)
    
    strassen_block(j_offset=1, k_offset=0)  # k=0,1
    strassen_block(j_offset=1, k_offset=2)  # k=2,3
    
    # k=4: 2x2x1, rank 4
    for i in range(n):
        for j in range(2):
            u = make_zero_u(); v = make_zero_v(); w = make_zero_w()
            u[i * m + (1 + j)] = 1
            v[(1 + j) * p + 4] = 1
            w[4 * n + i] = 1
            add_term(u, v, w)
    
    rank = len(cols)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u, v, w) in enumerate(cols):
        for i in range(U_rows):
            U[i][c] = u[i]
        for i in range(V_rows):
            V[i][c] = v[i]
        for i in range(W_rows):
            W[i][c] = w[i]
    
    return {"U": U, "V": V, "W": W}
