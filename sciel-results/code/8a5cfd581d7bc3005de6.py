"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Split m=5 into [2,2,1], p=6 into [2,2,2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Precompute offsets
    m_offsets = [0, 2, 4]  # start indices for each m-block
    p_offsets = [0, 2, 4]  # start indices for each p-block
    
    # We'll build U, V, W as lists of lists
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    U_rows = n * m  # 10
    V_rows = m * p  # 30
    W_rows = p * n  # 12
    
    # Start with empty lists, we'll append columns
    U_cols = [[] for _ in range(U_rows)]
    V_cols = [[] for _ in range(V_rows)]
    W_cols = [[] for _ in range(W_rows)]
    
    def add_column(u_vals, v_vals, w_vals):
        """Add a rank-1 term. u_vals: list of length U_rows, v_vals: length V_rows, w_vals: length W_rows"""
        for r in range(U_rows):
            U_cols[r].append(u_vals[r])
        for r in range(V_rows):
            V_cols[r].append(v_vals[r])
        for r in range(W_rows):
            W_cols[r].append(w_vals[r])
    
    def strassen_2x2x2(m_start, p_start):
        """Add Strassen decomposition for A[:, m_start:m_start+2] * B[m_start:m_start+2, p_start:p_start+2]
        
        A indices: rows i=0,1; cols j=m_start, m_start+1
        B indices: rows j=m_start, m_start+1; cols k=p_start, p_start+1
        C indices: rows i=0,1; cols k=p_start, p_start+1
        
        W convention: W[k*n+i, r]
        """
        # Local indices: i in {0,1}, j in {0,1} (local), k in {0,1} (local)
        # Global: i_global = i, j_global = m_start + j, k_global = p_start + k
        
        # Strassen 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # For each product Mr, we need U, V, W contributions
        # U[i_global, r] for i in {0,1}
        # V[j_global, r] for j in {m_start, m_start+1}
        # W[k_global*n + i_global, r] for k in {p_start, p_start+1}, i in {0,1}
        
        # Product 1: M1 = (A00+A11)(B00+B11)
        # A part: A[0, j0] + A[1, j1] where j0=m_start, j1=m_start+1
        # U: [1, 1] (for i=0,1)
        # V: [1, 1] (for j=j0, j1)
        # W: C00 += M1, C11 += M1
        # W[k*n+i]: for C00: k=p_start, i=0 -> W[p_start*2+0]; for C11: k=p_start+1, i=1 -> W[(p_start+1)*2+1]
        u1 = [0]*U_rows; u1[0] = 1; u1[1] = 1
        v1 = [0]*V_rows; v1[m_start] = 1; v1[m_start+1] = 1
        w1 = [0]*W_rows; w1[p_start*n + 0] = 1; w1[(p_start+1)*n + 1] = 1
        add_column(u1, v1, w1)
        
        # Product 2: M2 = (A10+A11)B00
        # A part: A[1, j0] + A[1, j1]
        # U: [0, 1] (only i=1)
        # V: [1, 0] (only j=j0)
        # W: C10 += M2 -> W[p_start*n + 1]
        u2 = [0]*U_rows; u2[1] = 1
        v2 = [0]*V_rows; v2[m_start] = 1
        w2 = [0]*W_rows; w2[p_start*n + 1] = 1
        add_column(u2, v2, w2)
        
        # Product 3: M3 = A00(B01-B11)
        # A part: A[0, j0]
        # U: [1, 0]
        # V: [0, 1] (B01 is at j=j0, k=p_start+1; B11 is at j=j1, k=p_start+1)
        # Wait, B01 means B[0,1] in local, which is B[j0, p_start+1]
        # B11 means B[1,1] in local, which is B[j1, p_start+1]
        # So V should have 1 at j0 and -1 at j1? No...
        # M3 = A[0,j0] * (B[j0, p_start+1] - B[j1, p_start+1])
        # This is a product of A[0,j0] and a linear combination of B entries
        # In tensor form: U[0] = 1, V[j0] = 1, V[j1] = -1, W for C01
        # C01 += M3 -> W[(p_start+1)*n + 0]
        u3 = [0]*U_rows; u3[0] = 1
        v3 = [0]*V_rows; v3[m_start] = 1; v3[m_start+1] = -1
        w3 = [0]*W_rows; w3[(p_start+1)*n + 0] = 1
        add_column(u3, v3, w3)
        
        # Product 4: M4 = A11(B10-B00)
        # A part: A[1, j1]
        # B part: B[j1, p_start] - B[j0, p_start]
        # U: [0, 1]
        # V: [0, 1] for j1, [-1, 0] for j0 -> V[j0]=-1, V[j1]=1
        # C10 += M4 -> W[p_start*n + 1]
        u4 = [0]*U_rows; u4[1] = 1
        v4 = [0]*V_rows; v4[m_start] = -1; v4[m_start+1] = 1
        w4 = [0]*W_rows; w4[p_start*n + 1] = 1
        add_column(u4, v4, w4)
        
        # Product 5: M5 = (A00+A01)B11
        # A part: A[0, j0] + A[0, j1]
        # U: [1, 0]
        # V: [0, 1] (only j1)
        # C00 -= M5, C01 += M5
        # W: C00: W[p_start*n + 0] -= M5, C01: W[(p_start+1)*n + 0] += M5
        u5 = [0]*U_rows; u5[0] = 1
        v5 = [0]*V_rows; v5[m_start+1] = 1
        w5 = [0]*W_rows; w5[p_start*n + 0] = -1; w5[(p_start+1)*n + 0] = 1
        add_column(u5, v5, w5)
        
        # Product 6: M6 = (A10-A00)(B00+B10)
        # A part: A[1, j0] - A[0, j0]
        # U: [-1, 1]
        # B part: B[j0, p_start] + B[j1, p_start]
        # V: [1, 1]
        # C11 += M6 -> W[(p_start+1)*n + 1]
        u6 = [0]*U_rows; u6[0] = -1; u6[1] = 1
        v6 = [0]*V_rows; v6[m_start] = 1; v6[m_start+1] = 1
        w6 = [0]*W_rows; w6[(p_start+1)*n + 1] = 1
        add_column(u6, v6, w6)
        
        # Product 7: M7 = (A01-A11)(B01+B11)
        # A part: A[0, j1] - A[1, j1]
        # U: [1, -1]
        # B part: B[j0, p_start+1] + B[j1, p_start+1]
        # V: [1, 1]
        # C00 += M7 -> W[p_start*n + 0]
        u7 = [0]*U_rows; u7[0] = 1; u7[1] = -1
        v7 = [0]*V_rows; v7[m_start] = 1; v7[m_start+1] = 1
        w7 = [0]*W_rows; w7[p_start*n + 0] = 1
        add_column(u7, v7, w7)
    
    def simple_2x1x2(m_start, p_start):
        """Add decomposition for A[:, m_start:m_start+1] * B[m_start:m_start+1, p_start:p_start+2]
        
        A: 2x1, B: 1x2, C: 2x2
        Rank 2 decomposition.
        """
        # C[i,k] = A[i, m_start] * B[m_start, p_start+k]
        # Term 1: A[0,m_start]*B[m_start,p_start] contributes to C[0,p_start]
        # A[0,m_start]*B[m_start,p_start+1] contributes to C[0,p_start+1]
        # A[1,m_start]*B[m_start,p_start] contributes to C[1,p_start]
        # A[1,m_start]*B[m_start,p_start+1] contributes to C[1,p_start+1]
        
        # This is outer product of (A[0,j], A[1,j]) and (B[j,k0], B[j,k1])
        # Rank 2: 
        # Term 1: U=[1,0], V=[1], W for C[0,k0] and C[0,k1]
        # Actually, let's use:
        # Term 1: U[0]=1, V[j]=1, W[k0*n+0]=1, W[k1*n+0]=1
        # Term 2: U[1]=1, V[j]=1, W[k0*n+1]=1, W[k1*n+1]=1
        
        u1 = [0]*U_rows; u1[0] = 1
        v1 = [0]*V_rows; v1[m_start] = 1
        w1 = [0]*W_rows; w1[p_start*n + 0] = 1; w1[(p_start+1)*n + 0] = 1
        add_column(u1, v1, w1)
        
        u2 = [0]*U_rows; u2[1] = 1
        v2 = [0]*V_rows; v2[m_start] = 1
        w2 = [0]*W_rows; w2[p_start*n + 1] = 1; w2[(p_start+1)*n + 1] = 1
        add_column(u2, v2, w2)
    
    # Process all blocks
    for mi, m_size in enumerate(m_splits):
        for pi, p_size in enumerate(p_splits):
            m_start = m_offsets[mi]
            p_start = p_offsets[pi]
            if m_size == 2 and p_size == 2:
                strassen_2x2x2(m_start, p_start)
            elif m_size == 1 and p_size == 2:
                simple_2x1x2(m_start, p_start)
            else:
                # Fallback to schoolbook for unexpected sizes
                for i in range(n):
                    for j in range(m_start, m_start + m_size):
                        for k in range(p_start, p_start + p_size):
                            u = [0]*U_rows; u[i] = 1
                            v = [0]*V_rows; v[j] = 1
                            w = [0]*W_rows; w[k*n + i] = 1
                            add_column(u, v, w)
    
    # Convert to row-major format: U[i][r], V[i][r], W[i][r]
    rank = len(U_cols[0])
    U = [[U_cols[i][r] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[i][r] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[i][r] for r in range(rank)] for i in range(W_rows)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
