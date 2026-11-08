# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 4 matrix by a 4 by 7 matrix.
    
    Strategy:
    1. Split the 4x4 core into two 2x2 blocks along the middle dimension.
       A (2x4) = [A1 | A2] where A1, A2 are 2x2.
       B (4x7) = [B1; B2] where B1, B2 are 2x7.
       C = A1*B1 + A2*B2.
       
    2. For each 2x2 * 2x7 multiplication, split the 7 columns of the right matrix
       into groups of size 2, 2, 2, 1.
       For 2x2 * 2x2, use Strassen's algorithm (rank 7).
       For 2x2 * 2x1, use schoolbook (rank 4).
       Total rank for one 2x2 * 2x7 is 7 + 7 + 7 + 4 = 25.
       
    3. Total rank for 2x4 * 4x7 is 25 + 25 = 50.
    """
    n, m, p = 2, 4, 7
    
    # Helper to generate Strassen decomposition for 2x2 * 2x2
    # Returns U, V, W parts for the sub-problem
    # Indices for U: 0..3 (2x2 matrix A)
    # Indices for V: 0..3 (2x2 matrix B)
    # Indices for W: 0..3 (2x2 matrix C)
    def strassen_2x2_2x2():
        # Standard Strassen
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1 - P4 + P6 - P2
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P3 + P5 - P7
        
        U = []
        V = []
        W = []
        
        # Term 1
        U.append([1, 0, 0, 1])
        V.append([1, 0, 0, 1])
        W.append([1, 0, 0, 1])
        
        # Term 2
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        W.append([-1, 0, 1, 0])
        
        # Term 3
        U.append([1, 0, 0, 0])
        V.append([0, 1, 0, -1])
        W.append([0, 1, 0, 0])
        
        # Term 4
        U.append([0, 0, 0, 1])
        V.append([0, 0, 1, -1])
        W.append([1, 0, 0, 1])
        
        # Term 5
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        W.append([0, 1, 0, -1])
        
        # Term 6
        U.append([0, 0, 1, -1])
        V.append([1, 1, 0, 0])
        W.append([0, 0, 1, 0])
        
        # Term 7
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        W.append([0, 0, 0, -1])
        
        return U, V, W

    # Helper to generate schoolbook decomposition for 2x2 * 2x1
    # A is 2x2, B is 2x1 (vector), C is 2x1 (vector)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    def schoolbook_2x2_2x1():
        U = []
        V = []
        W = []
        
        # C0 part
        # A00 * B0
        U.append([1, 0, 0, 0])
        V.append([1, 0])
        W.append([1, 0])
        
        # A01 * B1
        U.append([0, 1, 0, 0])
        V.append([0, 1])
        W.append([1, 0])
        
        # C1 part
        # A10 * B0
        U.append([0, 0, 1, 0])
        V.append([1, 0])
        W.append([0, 1])
        
        # A11 * B1
        U.append([0, 0, 0, 1])
        V.append([0, 1])
        W.append([0, 1])
        
        return U, V, W

    # We need to combine these into the global U, V, W matrices.
    # Global U size: 2*4 = 8
    # Global V size: 4*7 = 28
    # Global W size: 7*2 = 14
    
    # We will build the decomposition for one 2x2 * 2x7 block first.
    # Then replicate for the second block.
    
    # For a 2x2 * 2x7 block:
    # A is 2x2 (indices 0..3 in local U)
    # B is 2x7 (indices 0..13 in local V)
    # C is 2x7 (indices 0..13 in local W)
    
    # Split B's 7 columns into:
    # Col 0,1 -> 2x2 block (Strassen)
    # Col 2,3 -> 2x2 block (Strassen)
    # Col 4,5 -> 2x2 block (Strassen)
    # Col 6   -> 2x1 block (Schoolbook)
    
    def build_2x2_2x7_decomp():
        local_U = []
        local_V = []
        local_W = []
        
        # Block 1: Cols 0,1 (Strassen)
        U_s, V_s, W_s = strassen_2x2_2x2()
        for u, v, w in zip(U_s, V_s, W_s):
            # u is 4-dim (A)
            # v is 4-dim (B block) -> map to 14-dim (B full)
            # w is 4-dim (C block) -> map to 14-dim (C full)
            
            v_full = [0]*14
            w_full = [0]*14
            
            # B block indices: 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
            # Wait, my Strassen V indices are 0..3.
            # In 2x2 matrix B, index 0=B00, 1=B01, 2=B10, 3=B11?
            # Let's stick to row-major: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3.
            # In the 2x7 matrix, col 0 is B[0,0], B[1,0]. Col 1 is B[0,1], B[1,1].
            # So B[0,0] is V index 0*2+0 = 0? No, V index is j*p+k.
            # For local 2x2 * 2x7, j is 0..1, k is 0..6.
            # V index = j*7 + k.
            # B[0,0] -> j=0, k=0 -> 0
            # B[0,1] -> j=0, k=1 -> 1
            # B[1,0] -> j=1, k=0 -> 7
            # B[1,1] -> j=1, k=1 -> 8
            
            # My Strassen V vector [v0, v1, v2, v3] corresponds to B00, B01, B10, B11.
            # So v_full[0] += v[0]
            # v_full[1] += v[1]
            # v_full[7] += v[2]
            # v_full[8] += v[3]
            
            v_full[0] = v[0]
            v_full[1] = v[1]
            v_full[7] = v[2]
            v_full[8] = v[3]
            
            # C block indices: C[0,0], C[0,1], C[1,0], C[1,1]
            # W index = k*n + i. n=2.
            # C[0,0] -> k=0, i=0 -> 0
            # C[0,1] -> k=1, i=0 -> 2
            # C[1,0] -> k=0, i=1 -> 1
            # C[1,1] -> k=1, i=1 -> 3
            
            # My Strassen W vector [w0, w1, w2, w3] corresponds to C00, C01, C10, C11?
            # In strassen_2x2_2x2, W[0] is coeff for C00, W[1] for C01, W[2] for C10, W[3] for C11.
            # So w_full[0] += w[0]
            # w_full[2] += w[1]
            # w_full[1] += w[2]
            # w_full[3] += w[3]
            
            w_full[0] = w[0]
            w_full[2] = w[1]
            w_full[1] = w[2]
            w_full[3] = w[3]
            
            local_U.append(u)
            local_V.append(v_full)
            local_W.append(w_full)
            
        # Block 2: Cols 2,3 (Strassen)
        U_s, V_s, W_s = strassen_2x2_2x2()
        for u, v, w in zip(U_s, V_s, W_s):
            v_full = [0]*14
            w_full = [0]*14
            
            # Cols 2,3
            # B[0,2] -> 2, B[0,3] -> 3, B[1,2] -> 9, B[1,3] -> 10
            v_full[2] = v[0]
            v_full[3] = v[1]
            v_full[9] = v[2]
            v_full[10] = v[3]
            
            # C[0,2] -> 4, C[0,3] -> 5, C[1,2] -> 5? No.
            # W index = k*2 + i.
            # C[0,2]: k=2, i=0 -> 4
            # C[0,3]: k=3, i=0 -> 6
            # C[1,2]: k=2, i=1 -> 5
            # C[1,3]: k=3, i=1 -> 7
            
            w_full[4] = w[0]
            w_full[6] = w[1]
            w_full[5] = w[2]
            w_full[7] = w[3]
            
            local_U.append(u)
            local_V.append(v_full)
            local_W.append(w_full)
            
        # Block 3: Cols 4,5 (Strassen)
        U_s, V_s, W_s = strassen_2x2_2x2()
        for u, v, w in zip(U_s, V_s, W_s):
            v_full = [0]*14
            w_full = [0]*14
            
            # Cols 4,5
            # B[0,4] -> 4, B[0,5] -> 5, B[1,4] -> 11, B[1,5] -> 12
            v_full[4] = v[0]
            v_full[5] = v[1]
            v_full[11] = v[2]
            v_full[12] = v[3]
            
            # C[0,4] -> 8, C[0,5] -> 10, C[1,4] -> 9, C[1,5] -> 11
            w_full[8] = w[0]
            w_full[10] = w[1]
            w_full[9] = w[2]
            w_full[11] = w[3]
            
            local_U.append(u)
            local_V.append(v_full)
            local_W.append(w_full)
            
        # Block 4: Col 6 (Schoolbook 2x2 * 2x1)
        U_s, V_s, W_s = schoolbook_2x2_2x1()
        for u, v, w in zip(U_s, V_s, W_s):
            v_full = [0]*14
            w_full = [0]*14
            
            # Col 6
            # B[0,6] -> 6, B[1,6] -> 13
            v_full[6] = v[0]
            v_full[13] = v[1]
            
            # C[0,6] -> 12, C[1,6] -> 13
            w_full[12] = w[0]
            w_full[13] = w[1]
            
            local_U.append(u)
            local_V.append(v_full)
            local_W.append(w_full)
            
        return local_U, local_V, local_W

    # Now combine two 2x2 * 2x7 decompositions for the full 2x4 * 4x7.
    # A is 2x4. A1 is cols 0,1. A2 is cols 2,3.
    # B is 4x7. B1 is rows 0,1. B2 is rows 2,3.
    
    U_1, V_1, W_1 = build_2x2_2x7_decomp()
    U_2, V_2, W_2 = build_2x2_2x7_decomp()
    
    global_U = []
    global_V = []
    global_W = []
    
    # Add decomposition for A1 * B1
    for u, v, w in zip(U_1, V_1, W_1):
        # u is 4-dim (A1). Map to 8-dim (A full).
        # A1 indices: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1].
        # Global A index: i*m + j. m=4.
        # A[0,0] -> 0, A[0,1] -> 1, A[1,0] -> 4, A[1,1] -> 5.
        
        u_full = [0]*8
        u_full[0] = u[0]
        u_full[1] = u[1]
        u_full[4] = u[2]
        u_full[5] = u[3]
        
        # v is 14-dim (B1). Map to 28-dim (B full).
        # B1 indices: j=0,1.
        # Global B index: j*p + k. p=7.
        # B[0,k] -> k. B[1,k] -> 7+k.
        # Local V index: j*7+k.
        # So local index l corresponds to global index l.
        # Wait, local V is 2x7. j in 0..1.
        # Local index = j*7 + k.
        # Global index = j*7 + k.
        # They are the same!
        
        global_U.append(u_full)
        global_V.append(v)
        global_W.append(w)
        
    # Add decomposition for A2 * B2
    for u, v, w in zip(U_2, V_2, W_2):
        # u is 4-dim (A2). Map to 8-dim (A full).
        # A2 indices: 0,1,2,3 correspond to A[0,2], A[0,3], A[1,2], A[1,3].
        # Global A index:
        # A[0,2] -> 2, A[0,3] -> 3, A[1,2] -> 6, A[1,3] -> 7.
        
        u_full = [0]*8
        u_full[2] = u[0]
        u_full[3] = u[1]
        u_full[6] = u[2]
        u_full[7] = u[3]
        
        # v is 14-dim (B2). Map to 28-dim (B full).
        # B2 indices: j=2,3.
        # Global B index: j*7 + k.
        # B[2,k] -> 14+k. B[3,k] -> 21+k.
        # Local V index: j_local*7 + k, where j_local is 0..1.
        # j_local=0 corresponds to global j=2.
        # j_local=1 corresponds to global j=3.
        
        v_full = [0]*28
        for l in range(14):
            j_local = l // 7
            k = l % 7
            j_global = 2 + j_local
            global_idx = j_global * 7 + k
            v_full[global_idx] = v[l]
            
        global_U.append(u_full)
        global_V.append(v_full)
        global_W.append(w)
        
    return {"U": global_U, "V": global_V, "W": global_W}
# EVOLVE-BLOCK-END
