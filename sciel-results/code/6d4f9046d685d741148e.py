import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # 1. Define base Strassen 2x2x2
    # Returns U, V, W for 2x2x2 with rank 7
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1 # C11
        W[3, 0] = 1 # C22
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1 # C21
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1 # C12
        
        # M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1 # C11
        W[1, 3] = 1 # C21
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C11
        W[3, 4] = 1  # C22
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C22
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C11
        W[2, 6] = 1  # C12
        
        return U, V, W

    # 2. Build 4x4x4 using Strassen
    # We will build the 49 terms.
    # The "outer" Strassen defines linear combinations of 2x2 blocks.
    # Let's define the outer coefficients.
    # A blocks: A00, A01, A10, A11 (indices 0,1,2,3 in block space)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Outer M1 = (A00+A11)(B00+B11)
    # Outer M2 = (A10+A11)B00
    # Outer M3 = A00(B01-B11)
    # Outer M4 = A11(B10-B00)
    # Outer M5 = (A00+A01)B11
    # Outer M6 = (A10-A00)(B00+B01)
    # Outer M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each outer term r, we have U_A (4x1 vec of block coeffs), V_B (4x1 vec), W_C (4x1 vec).
    # Actually, U_A is a vector of length 4 (coeffs for A00, A01, A10, A11).
    # But A is flattened as A00, A01, A10, A11?
    # A is 4x4. 
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # Flat index of A:
    # 0: A00[0,0], 1: A00[0,1], 2: A00[1,0], 3: A00[1,1]
    # 4: A01[0,0], ...
    # This is getting complicated to map directly.
    
    # Easier: Use Kronecker products.
    # The 4x4x4 Strassen tensor is the "substitution" of the 2x2x2 Strassen into itself.
    # We can generate the 49 rank-1 tensors by iterating over the 7 outer terms and 7 inner terms.
    
    U_4x4x4 = []
    V_4x4x4 = []
    W_4x4x4 = []
    
    # Get base 2x2x2 tensors
    U_base, V_base, W_base = strassen_2x2x2()
    
    # Define outer coefficients
    # u_outer[r] is a 4x1 vector representing coeffs of A blocks (A00, A01, A10, A11)
    # v_outer[r] is a 4x1 vector representing coeffs of B blocks
    # w_outer[r] is a 4x1 vector representing coeffs of C blocks
    
    # Note: Block order in flat 4x4 matrix:
    # A = [ A00 A01 ]
    #     [ A10 A11 ]
    # Flat A: A00(4 elems), A01(4 elems), A10(4 elems), A11(4 elems)
    # So if we have a coeff `c` for block A00, it adds `c` to the first 4 elements of U.
    
    u_outer = []
    v_outer = []
    w_outer = []
    
    # M1: (A00+A11)(B00+B11)
    u_outer.append([1, 0, 0, 1]) # A00, A01, A10, A11
    v_outer.append([1, 0, 0, 1])
    w_outer.append([1, 0, 0, 1]) # C00, C01, C10, C11? No.
    # C = [ C00 C01 ]
    #     [ C10 C11 ]
    # Flat C: C00(4), C01(4), C10(4), C11(4)
    # W index is k*n+i.
    # C00 is top-left. C01 is top-right. C10 is bottom-left. C11 is bottom-right.
    # In W (p x n = 4x4), index k*4+i.
    # C00 (i=0,1; k=0,1) -> W[0:4]
    # C01 (i=0,1; k=2,3) -> W[4:8]? No.
    # W shape is (p*n, rank) = (16, rank).
    # W index = k*n + i.
    # C00: i in {0,1}, k in {0,1}. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
    # C01: i in {0,1}, k in {2,3}. Indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13.
    # C10: i in {2,3}, k in {0,1}. Indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
    # C11: i in {2,3}, k in {2,3}. Indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15.
    
    # So w_outer should be a 16x1 vector? Or we can just build the full 16x1 vectors directly.
    
    # Let's build the full 16-length vectors for U, V, W for each of the 49 terms.
    
    for r in range(7):
        # Determine coeffs for A, B, C blocks for this outer term
        if r == 0:
            u_blocks = [1, 0, 0, 1]
            v_blocks = [1, 0, 0, 1]
            w_blocks = [1, 0, 0, 1] # C00, C01, C10, C11 coeffs? 
            # M1 contributes to C00 and C11.
            # C00 = M1 + ...
            # C11 = M1 + ...
            # So w_blocks for M1 should have 1 for C00 and 1 for C11.
            # w_blocks = [1, 0, 0, 1] means C00=1, C01=0, C10=0, C11=1. Correct.
        elif r == 1:
            u_blocks = [0, 0, 1, 1] # A10, A11
            v_blocks = [1, 0, 0, 0] # B00
            w_blocks = [0, 0, 1, 0] # C10
        elif r == 2:
            u_blocks = [1, 0, 0, 0] # A00
            v_blocks = [0, 1, 0, -1] # B01 - B11
            w_blocks = [0, 1, 0, 0] # C01
        elif r == 3:
            u_blocks = [0, 0, 0, 1] # A11
            v_blocks = [0, 0, 1, -1] # B10 - B00
            w_blocks = [1, 0, 1, 0] # C00 + C10
        elif r == 4:
            u_blocks = [1, 1, 0, 0] # A00 + A01
            v_blocks = [0, 0, 0, 1] # B11
            w_blocks = [-1, 0, 0, 1] # -C00 + C11
        elif r == 5:
            u_blocks = [0, 0, 1, -1] # A10 - A00
            v_blocks = [1, 1, 0, 0] # B00 + B01
            w_blocks = [0, 0, 0, 1] # C11
        elif r == 6:
            u_blocks = [0, 1, 0, -1] # A01 - A11
            v_blocks = [0, 0, 1, 1] # B10 + B11
            w_blocks = [1, 0, 0, 0] # C00
        else:
            pass
            
        # Now expand each block coeff into 4 scalar coeffs using base Strassen
        # For each of the 7 inner terms s:
        for s in range(7):
            # u_vec is 16x1
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            # Fill u_vec
            # u_blocks[0] corresponds to A00 (indices 0-3)
            # u_blocks[1] corresponds to A01 (indices 4-7)
            # u_blocks[2] corresponds to A10 (indices 8-11)
            # u_blocks[3] corresponds to A11 (indices 12-15)
            
            # Base U for this inner term s is U_base[:, s] (length 4)
            # U_base indices correspond to a,b,c,d of the 2x2 matrix
            # a->0, b->1, c->2, d->3
            
            for block_idx in range(4):
                coeff = u_blocks[block_idx]
                if coeff != 0:
                    # Add coeff * U_base[:, s] to the corresponding slice
                    start = block_idx * 4
                    u_vec[start:start+4] += coeff * U_base[:, s]
                    
            # Fill v_vec
            for block_idx in range(4):
                coeff = v_blocks[block_idx]
                if coeff != 0:
                    start = block_idx * 4
                    v_vec[start:start+4] += coeff * V_base[:, s]
                    
            # Fill w_vec
            # w_blocks[0] corresponds to C00.
            # C00 indices in W (k*4+i): 0,1,4,5
            # C01 indices: 8,9,12,13
            # C10 indices: 2,3,6,7
            # C11 indices: 10,11,14,15
            
            # Base W for inner term s is W_base[:, s] (length 4)
            # W_base indices correspond to C11, C21, C12, C22 of the 2x2 block?
            # In strassen_2x2x2, W indices were:
            # 0: C11 (top-left of 2x2)
            # 1: C21 (bottom-left)
            # 2: C12 (top-right)
            # 3: C22 (bottom-right)
            
            # So for C00 block (top-left of 4x4), the local 2x2 indices map to global W indices:
            # local 0 (TL) -> global 0
            # local 1 (BL) -> global 4? No.
            # Let's re-verify W indexing for 4x4.
            # W[k*n+i].
            # C00 block: i in {0,1}, k in {0,1}.
            # Local 2x2: i_l in {0,1}, k_l in {0,1}.
            # Global i = i_l. Global k = k_l.
            # Global idx = k_l * 4 + i_l.
            # local 0 (i_l=0, k_l=0) -> 0
            # local 1 (i_l=1, k_l=0) -> 4? No, k_l=0, i_l=1 -> 1.
            # Wait, in strassen_2x2x2, W index was k*2+i.
            # local 0: k=0,i=0 -> 0
            # local 1: k=0,i=1 -> 1
            # local 2: k=1,i=0 -> 2
            # local 3: k=1,i=1 -> 3
            
            # So for C00 block:
            # local 0 -> global 0*4+0 = 0
            # local 1 -> global 0*4+1 = 1
            # local 2 -> global 1*4+0 = 4
            # local 3 -> global 1*4+1 = 5
            
            # For C01 block (top-right): i in {0,1}, k in {2,3}.
            # local 0 (i_l=0, k_l=0) -> global k=2, i=0 -> 8
            # local 1 (i_l=1, k_l=0) -> global k=2, i=1 -> 9
            # local 2 (i_l=0, k_l=1) -> global k=3, i=0 -> 12
            # local 3 (i_l=1, k_l=1) -> global k=3, i=1 -> 13
            
            # For C10 block (bottom-left): i in {2,3}, k in {0,1}.
            # local 0 -> global k=0, i=2 -> 2
            # local 1 -> global k=0, i=3 -> 3
            # local 2 -> global k=1, i=2 -> 6
            # local 3 -> global k=1, i=3 -> 7
            
            # For C11 block (bottom-right): i in {2,3}, k in {2,3}.
            # local 0 -> global k=2, i=2 -> 10
            # local 1 -> global k=2, i=3 -> 11
            # local 2 -> global k=3, i=2 -> 14
            # local 3 -> global k=3, i=3 -> 15
            
            c00_map = [0, 1, 4, 5]
            c01_map = [8, 9, 12, 13]
            c10_map = [2, 3, 6, 7]
            c11_map = [10, 11, 14, 15]
            
            maps = [c00_map, c01_map, c10_map, c11_map]
            
            for block_idx in range(4):
                coeff = w_blocks[block_idx]
                if coeff != 0:
                    m = maps[block_idx]
                    w_vec[m[0]] += coeff * W_base[0, s]
                    w_vec[m[1]] += coeff * W_base[1, s]
                    w_vec[m[2]] += coeff * W_base[2, s]
                    w_vec[m[3]] += coeff * W_base[3, s]
                    
            U_4x4x4.append(u_vec)
            V_4x4x4.append(v_vec)
            W_4x4x4.append(w_vec)
            
    # Convert to matrices
    U_4x4x4 = np.array(U_4x4x4).T # 16x49
    V_4x4x4 = np.array(V_4x4x4).T # 16x49
    W_4x4x4 = np.array(W_4x4x4).T # 16x49
    
    # 3. Construct full 4x4x9
    # Part 1: 4x4x4 (cols 0-3 of B)
    # Part 2: 4x4x4 (cols 4-7 of B)
    # Part 3: 4x4x1 (col 8 of B)
    
    # Part 3: Schoolbook 4x4x1
    U_part3 = []
    V_part3 = []
    W_part3 = []
    for i in range(4):
        for j in range(4):
            u = np.zeros(16); u[i*4+j] = 1
            v = np.zeros(4); v[j*1+0] = 1 # j*p+k, p=1, k=0
            w = np.zeros(4); w[0*4+i] = 1 # k*n+i, k=0
            U_part3.append(u)
            V_part3.append(v)
            W_part3.append(w)
            
    U_part3 = np.array(U_part3).T # 16x16
    V_part3 = np.array(V_part3).T # 4x16
    W_part3 = np.array(W_part3).T # 4x16
    
    # Assemble full U (16x114)
    U_full = np.zeros((16, 114))
    U_full[:, 0:49] = U_4x4x4
    U_full[:, 49:98] = U_4x4x4
    U_full[:, 98:114] = U_part3
    
    # Assemble full V (36x114)
    V_full = np.zeros((36, 114))
    # Part 1: B cols 0-3. V indices j*9+k.
    # j in 0..3, k in 0..3.
    # For each col r in 0..48:
    #   v_vec is 16x1 from V_4x4x4[:, r]
    #   We need to place these into V_full[?, r]
    #   V_4x4x4 index is j*4+k (for 4x4 B).
    #   V_full index is j*9+k.
    for r in range(49):
        v_vec = V_4x4x4[:, r]
        for j in range(4):
            for k in range(4):
                V_full[j*9+k, r] = v_vec[j*4+k]
                
    # Part 2: B cols 4-7.
    for r in range(49):
        v_vec = V_4x4x4[:, r]
        for j in range(4):
            for k in range(4):
                # k in 4..7 corresponds to local k 0..3
                V_full[j*9+(k+4), r+49] = v_vec[j*4+k]
                
    # Part 3: B col 8.
    for r in range(16):
        v_vec = V_part3[:, r]
        for j in range(4):
            # k=0 corresponds to global k=8
            V_full[j*9+8, r+98] = v_vec[j*1+0]
            
    # Assemble full W (36x114)
    W_full = np.zeros((36, 114))
    # Part 1: C cols 0-3. W indices k*4+i.
    for r in range(49):
        w_vec = W_4x4x4[:, r]
        for k in range(4):
            for i in range(4):
                W_full[k*4+i, r] = w_vec[k*4+i]
                
    # Part 2: C cols 4-7.
    for r in range(49):
        w_vec = W_4x4x4[:, r]
        for k in range(4):
            for i in range(4):
                # k in 4..7 corresponds to local k 0..3
                W_full[(k+4)*4+i, r+49] = w_vec[k*4+i]
                
    # Part 3: C col 8.
    for r in range(16):
        w_vec = W_part3[:, r]
        for k in range(1):
            for i in range(4):
                W_full[8*4+i, r+98] = w_vec[k*4+i]
                
    return {"U": U_full, "V": V_full, "W": W_full}
