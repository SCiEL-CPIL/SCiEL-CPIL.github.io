import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions: A is n x m, B is m x p, C is n x p
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Indexing convention:
    # U[i*m + j] corresponds to A[i, j]
    # V[j*p + k] corresponds to B[j, k]
    # W[k*n + i] corresponds to C[i, k]
    
    # We will decompose the matrix multiplication into blocks.
    # A is 2x4, B is 4x5.
    # We can split the middle dimension m=4 into two blocks of size 2.
    # Let A = [A_L | A_R] where A_L is 2x2 and A_R is 2x2.
    # Let B = [B_T; B_B] where B_T is 2x5 and B_B is 2x5.
    # C = A_L * B_T + A_R * B_B.
    
    # For each 2x2 * 2x5 multiplication, we can use a rank-7 decomposition for the 2x2 part
    # and then extend it to 2x5.
    # A 2x2 matrix multiplication can be done in rank 7 (Strassen).
    # A 2x2 * 2x5 multiplication:
    # Let A2 be 2x2 and B25 be 2x5.
    # C25 = A2 * B25.
    # We can view B25 as two 2x2 blocks and one 2x1 block? Or just use the tensor decomposition.
    # The tensor for 2x2*2x5 has dimension 2*2*5 = 20.
    # The rank of 2x2*2x5 is known to be 14? Let's check.
    # Actually, we can just use the trivial decomposition for the 2x5 part if we don't have a better one.
    # Trivial rank for 2x2*2x5 is 2*2*5 = 20.
    # Strassen for 2x2*2x2 is rank 7.
    # So for 2x2*2x5, we can split the 5 columns into 2+2+1.
    # Rank = 7 (for first 2 cols) + 7 (for next 2 cols) + 4 (for last 1 col, trivial 2x2*1 is 2*2*1=4? No, 2x1*2x1 is rank 2? No, 2x2*1x1? No.
    # A is 2x2, B is 2x1. C is 2x1.
    # Trivial rank is 2*2*1 = 4.
    # So total rank for 2x2*2x5 using Strassen blocks: 7 + 7 + 4 = 18.
    # Is there a better decomposition for 2x2*2x5?
    # The rank of the 2x2*2x5 tensor is actually 14?
    # Let's stick to the block decomposition which is safer.
    # Total rank = 2 * (7 + 7 + 4) = 2 * 18 = 36.
    # This is better than 40.
    
    # Let's verify the rank of 2x2*2x1.
    # A (2x2) * B (2x1) = C (2x1).
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # This is a bilinear map.
    # We can write it as:
    # M1 = A00*B0 -> C0 += M1
    # M2 = A01*B1 -> C0 += M2
    # M3 = A10*B0 -> C1 += M3
    # M4 = A11*B1 -> C1 += M4
    # Rank 4 is trivial.
    # Can we do it in rank 3?
    # Strassen doesn't apply directly to 2x2*2x1.
    # However, the rank of 2x2*2x1 is known to be 4?
    # Actually, for 2x2*2x1, the tensor is isomorphic to the tensor for 2x1*2x2.
    # The rank of 2x2*2x1 is 4.
    
    # So, plan:
    # 1. Define the Strassen decomposition for 2x2*2x2 (rank 7).
    # 2. Define the trivial decomposition for 2x2*2x1 (rank 4).
    # 3. Combine them for 2x2*2x5 (rank 18).
    # 4. Apply this to the two blocks of the 2x4*4x5 problem.
    # Total rank = 18 + 18 = 36.
    
    def strassen_2x2_2x2():
        """
        Returns U, V, W for 2x2*2x2 multiplication.
        U, V, W are lists of vectors.
        U[i*2+j] for A[i,j]
        V[j*2+k] for B[j,k]
        W[k*2+i] for C[i,k]
        """
        # Strassen formulas:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B21 - B22)
        # M4 = A22 * (B12 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = []
        V = []
        W = []
        
        # Helper to add a term
        def add_term(u_vec, v_vec, w_vec):
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            
        # Indices for 2x2:
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (W index is k*2+i)
        # C00 -> 0, C01 -> 1, C10 -> 2, C11 -> 3
        
        # M1
        u = [1, 0, 0, 1] # A00 + A11
        v = [1, 0, 0, 1] # B00 + B11
        w = [1, 0, 0, 1] # C00 + C11
        add_term(u, v, w)
        
        # M2
        u = [0, 0, 1, 1] # A10 + A11
        v = [1, 0, 0, 0] # B00
        w = [0, 0, 1, 0] # C10
        add_term(u, v, w)
        
        # M3
        u = [1, 0, 0, 0] # A00
        v = [0, 0, 1, -1] # B10 - B11
        w = [0, 1, 0, 0] # C01
        add_term(u, v, w)
        
        # M4
        u = [0, 0, 0, 1] # A11
        v = [0, 1, -1, 0] # B01 - B00
        w = [1, 0, 0, 0] # C00
        add_term(u, v, w)
        
        # M5
        u = [1, 1, 0, 0] # A00 + A01
        v = [0, 0, 0, 1] # B11
        w = [-1, 0, 0, 0] # -C00
        add_term(u, v, w)
        
        # M6
        u = [0, 0, 1, -1] # A10 - A00
        v = [1, 1, 0, 0] # B00 + B01
        w = [0, 0, 0, 1] # C11
        add_term(u, v, w)
        
        # M7
        u = [0, 1, 0, -1] # A01 - A11
        v = [0, 0, 1, 1] # B10 + B11
        w = [1, 0, 0, 0] # C00
        add_term(u, v, w)
        
        return U, V, W

    def trivial_2x2_2x1():
        """
        Returns U, V, W for 2x2*2x1 multiplication.
        U[i*2+j] for A[i,j]
        V[j*1+k] for B[j,k] (k is always 0, so V[j] for B[j,0])
        W[k*2+i] for C[i,k] (k is always 0, so W[i] for C[i,0])
        """
        U = []
        V = []
        W = []
        
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # Term 1: A00 * B0 -> C0
        add_term([1, 0, 0, 0], [1, 0], [1, 0])
        
        # Term 2: A01 * B1 -> C0
        add_term([0, 1, 0, 0], [0, 1], [1, 0])
        
        # Term 3: A10 * B0 -> C1
        add_term([0, 0, 1, 0], [1, 0], [0, 1])
        
        # Term 4: A11 * B1 -> C1
        add_term([0, 0, 0, 1], [0, 1], [0, 1])
        
        return U, V, W

    # Now we combine these to form the 2x4*4x5 decomposition.
    # A is 2x4. Split into A_L (2x2) and A_R (2x2).
    # B is 4x5. Split into B_T (2x5) and B_B (2x5).
    # C = A_L * B_T + A_R * B_B.
    
    # We need to map the local indices to global indices.
    # Global U index: i*m + j. i in 0..1, j in 0..3.
    # Global V index: j*p + k. j in 0..3, k in 0..4.
    # Global W index: k*n + i. k in 0..4, i in 0..1.
    
    # Block 1: A_L (j=0,1) * B_T (j=0,1, k=0..4)
    # Block 2: A_R (j=2,3) * B_B (j=2,3, k=0..4)
    
    # For Block 1:
    # A_L indices: 0,1,2,3 (local) -> Global: 0,1,2,3
    # B_T indices: j in 0,1, k in 0..4.
    # We split B_T into 3 parts:
    # Part 1: k=0,1 (2x2) -> Use Strassen
    # Part 2: k=2,3 (2x2) -> Use Strassen
    # Part 3: k=4 (2x1) -> Use Trivial
    
    # Strassen U, V, W are for 2x2*2x2.
    # We need to expand them to the full dimensions.
    
    def expand_strassen(U_s, V_s, W_s, a_offset, b_row_offset, b_col_offset):
        """
        Expands a 2x2*2x2 decomposition to the global 2x4*4x5 context.
        a_offset: row offset for A (always 0 for 2 rows)
        b_row_offset: row offset for B (0 or 2)
        b_col_offset: col offset for B (0, 2, or 4)
        """
        U_g = []
        V_g = []
        W_g = []
        
        for u, v, w in zip(U_s, V_s, W_s):
            # Expand U (size 4) to size 8 (2*4)
            u_g = [0] * 8
            for idx, val in enumerate(u):
                i, j = divmod(idx, 2)
                u_g[i * 4 + j] = val # A[i, j]
            
            # Expand V (size 4) to size 20 (4*5)
            v_g = [0] * 20
            for idx, val in enumerate(v):
                j, k = divmod(idx, 2)
                v_g[(j + b_row_offset) * 5 + (k + b_col_offset)] = val # B[j+b_row_offset, k+b_col_offset]
            
            # Expand W (size 4) to size 10 (5*2)
            w_g = [0] * 10
            for idx, val in enumerate(w):
                k, i = divmod(idx, 2)
                w_g[(k + b_col_offset) * 2 + i] = val # C[i, k+b_col_offset]
            
            U_g.append(u_g)
            V_g.append(v_g)
            W_g.append(w_g)
        
        return U_g, V_g, W_g

    def expand_trivial_2x2_2x1(U_t, V_t, W_t, a_offset, b_row_offset, b_col_offset):
        """
        Expands a 2x2*2x1 decomposition to the global 2x4*4x5 context.
        """
        U_g = []
        V_g = []
        W_g = []
        
        for u, v, w in zip(U_t, V_t, W_t):
            # Expand U (size 4) to size 8
            u_g = [0] * 8
            for idx, val in enumerate(u):
                i, j = divmod(idx, 2)
                u_g[i * 4 + j] = val
            
            # Expand V (size 2) to size 20
            v_g = [0] * 20
            for idx, val in enumerate(v):
                j = idx
                v_g[(j + b_row_offset) * 5 + b_col_offset] = val # B[j+b_row_offset, b_col_offset]
            
            # Expand W (size 2) to size 10
            w_g = [0] * 10
            for idx, val in enumerate(w):
                i = idx
                w_g[b_col_offset * 2 + i] = val # C[i, b_col_offset]
            
            U_g.append(u_g)
            V_g.append(v_g)
            W_g.append(w_g)
        
        return U_g, V_g, W_g

    U_total = []
    V_total = []
    W_total = []
    
    # Get base decompositions
    U_s, V_s, W_s = strassen_2x2_2x2()
    U_t, V_t, W_t = trivial_2x2_2x1()
    
    # Block 1: A_L (j=0,1) * B_T (j=0,1)
    # Part 1: k=0,1
    U_g1, V_g1, W_g1 = expand_strassen(U_s, V_s, W_s, 0, 0, 0)
    U_total.extend(U_g1)
    V_total.extend(V_g1)
    W_total.extend(W_g1)
    
    # Part 2: k=2,3
    U_g2, V_g2, W_g2 = expand_strassen(U_s, V_s, W_s, 0, 0, 2)
    U_total.extend(U_g2)
    V_total.extend(V_g2)
    W_total.extend(W_g2)
    
    # Part 3: k=4
    U_g3, V_g3, W_g3 = expand_trivial_2x2_2x1(U_t, V_t, W_t, 0, 0, 4)
    U_total.extend(U_g3)
    V_total.extend(V_g3)
    W_total.extend(W_g3)
    
    # Block 2: A_R (j=2,3) * B_B (j=2,3)
    # Part 1: k=0,1
    U_g4, V_g4, W_g4 = expand_strassen(U_s, V_s, W_s, 0, 2, 0)
    U_total.extend(U_g4)
    V_total.extend(V_g4)
    W_total.extend(W_g4)
    
    # Part 2: k=2,3
    U_g5, V_g5, W_g5 = expand_strassen(U_s, V_s, W_s, 0, 2, 2)
    U_total.extend(U_g5)
    V_total.extend(V_g5)
    W_total.extend(W_g5)
    
    # Part 3: k=4
    U_g6, V_g6, W_g6 = expand_trivial_2x2_2x1(U_t, V_t, W_t, 0, 2, 4)
    U_total.extend(U_g6)
    V_total.extend(V_g6)
    W_total.extend(W_g6)
    
    rank = len(U_total)
    
    return {"U": U_total, "V": V_total, "W": W_total}
