import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    # We will build the decomposition block by block.
    # We maintain a list of columns for U, V, W.
    # Each column corresponds to a rank-1 term in the decomposition.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomposition(U_local, V_local, W_local, i_offset, j_offset, k_offset, i_dim, j_dim, k_dim):
        """
        Adds a local decomposition to the global lists.
        U_local: (i_dim * j_dim, rank)
        V_local: (j_dim * k_dim, rank)
        W_local: (k_dim * i_dim, rank)
        i_offset, j_offset, k_offset: start indices in the global dimensions.
        """
        rank = U_local.shape[1]
        for r in range(rank):
            u_col = np.zeros(nm)
            v_col = np.zeros(mp)
            w_col = np.zeros(pn)
            
            # Map U indices
            # Local U index: i_l * j_dim + j_l
            # Global U index: (i_l + i_offset) * m + (j_l + j_offset)
            for i_l in range(i_dim):
                for j_l in range(j_dim):
                    val = U_local[i_l * j_dim + j_l, r]
                    if val != 0:
                        u_col[(i_l + i_offset) * m + (j_l + j_offset)] = val
            
            # Map V indices
            # Local V index: j_l * k_dim + k_l
            # Global V index: (j_l + j_offset) * p + (k_l + k_offset)
            for j_l in range(j_dim):
                for k_l in range(k_dim):
                    val = V_local[j_l * k_dim + k_l, r]
                    if val != 0:
                        v_col[(j_l + j_offset) * p + (k_l + k_offset)] = val
            
            # Map W indices
            # Local W index: k_l * i_dim + i_l
            # Global W index: (k_l + k_offset) * n + (i_l + i_offset)
            for k_l in range(k_dim):
                for i_l in range(i_dim):
                    val = W_local[k_l * i_dim + i_l, r]
                    if val != 0:
                        w_col[(k_l + k_offset) * n + (i_l + i_offset)] = val
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # Base Decompositions
    
    def get_strassen_222():
        # Returns U, V, W for 2x2x2 multiplication with rank 7
        # C = AB, A(2,2), B(2,2)
        # Strassen formulas:
        # m1 = (A11 + A22)(B11 + B22)
        # m2 = (A21 + A22)B11
        # m3 = A11(B12 - B22)
        # m4 = A22(B21 - B11)
        # m5 = (A11 + A12)B22
        # m6 = (A21 - A11)(B11 + B12)
        # m7 = (A12 - A22)(B21 + B22)
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1 # C11, C22? No, W index is k*n+i.
        # C11 is i=0,k=0. C22 is i=1,k=1.
        # W[0] is k=0,i=0 (C11). W[1] is k=0,i=1 (C21). W[2] is k=1,i=0 (C12). W[3] is k=1,i=1 (C22).
        W[0,0] = 1
        W[3,0] = 1
        
        # m2
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1 # C21
        W[3,1] = 1 # C22 (with -1)
        W[3,1] = -1
        
        # m3
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[2,2] = 1 # C12
        W[3,2] = 1 # C22
        
        # m4
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1 # C11
        W[1,3] = 1 # C21
        
        # m5
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1 # C11
        W[2,4] = 1 # C12
        
        # m6
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1 # C22
        
        # m7
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1 # C11
        W[2,6] = -1 # C12
        
        return U, V, W

    def get_schoolbook(n_loc, m_loc, p_loc):
        U = np.zeros((n_loc * m_loc, n_loc * m_loc * p_loc))
        V = np.zeros((m_loc * p_loc, n_loc * m_loc * p_loc))
        W = np.zeros((p_loc * n_loc, n_loc * m_loc * p_loc))
        col = 0
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    U[i * m_loc + j, col] = 1
                    V[j * p_loc + k, col] = 1
                    W[k * n_loc + i, col] = 1
                    col += 1
        return U, V, W

    # Composite Decompositions
    
    def get_223():
        # 2x2 * 2x3 = 2x3
        # Split B into B1 (2x2) and B2 (2x1)
        # A B1 (2x2x2, rank 7)
        # A B2 (2x2x1, rank 4)
        # Total rank 11
        
        U1, V1, W1 = get_strassen_222()
        # U1: (4,7), V1: (4,7), W1: (4,7)
        # W1 indices: k*n+i. k in 0,1. n=2.
        
        U2, V2, W2 = get_schoolbook(2, 2, 1)
        # U2: (4,4), V2: (2,4), W2: (2,4)
        
        # We need to map these to the 2x2x3 space.
        # U is (2*2, R).
        # V is (2*3, R).
        # W is (3*2, R).
        
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R = R1 + R2
        
        U = np.zeros((4, R))
        V = np.zeros((6, R))
        W = np.zeros((6, R))
        
        # Part 1: A B1
        U[:, :R1] = U1
        # V1 is (4, R1). Indices j*p+k. p=2.
        # We need V to be (6, R). Indices j*3+k.
        # j in 0,1. k in 0,1.
        for j in range(2):
            for k in range(2):
                V[j*3 + k, :R1] = V1[j*2 + k, :R1]
        # W1 is (4, R1). Indices k*n+i. n=2.
        # We need W to be (6, R). Indices k*2+i.
        # k in 0,1. i in 0,1.
        W[:, :R1] = W1
        
        # Part 2: A B2
        U[:, R1:] = U2
        # V2 is (2, R2). Indices j*1+k. j in 0,1. k=0.
        # We need V to be (6, R). Indices j*3+k.
        # B2 corresponds to columns 2 of B (index 2 in 0..2).
        # So j in 0,1. k=2.
        for j in range(2):
            V[j*3 + 2, R1:] = V2[j*1 + 0, R1:]
        # W2 is (2, R2). Indices k*2+i. k=0.
        # We need W to be (6, R). Indices k*2+i.
        # B2 produces C columns? No, B2 is the second part of B.
        # Wait, A(2x2) * B2(2x1) gives C_part(2x1).
        # This C_part corresponds to the 3rd column of C (index 2).
        # So k=2.
        # W indices are k*n+i.
        # So we place W2 at k=2.
        W[2*2 + 0:2*2 + 2, R1:] = W2[0:2, R1:]
        
        return U, V, W

    def get_233():
        # 2x3 * 3x3 = 2x3
        # Split A into A1 (2x2) and A2 (2x1)
        # A1 B (2x2 * 2x3? No, B is 3x3. A1 is 2x2. B needs to be 2x3? No.)
        # A(2x3) * B(3x3).
        # Split A: A1 (2x2, cols 0,1), A2 (2x1, col 2).
        # A1 * B[:, 0:2] (2x2 * 2x3? No, B[:, 0:2] is 3x2. A1 is 2x2. Product is 2x2.)
        # This is getting messy.
        # Let's use schoolbook for 2x3x3 to be safe? Rank 18.
        # Or 17 as derived?
        # A1 (2x2) * B1 (2x3) where B1 = B[:, 0:2]? No, B is 3x3.
        # A1 (2x2) * B_top (2x3) where B_top = B[0:2, :].
        # A2 (2x1) * B_bot (1x3) where B_bot = B[2, :].
        
        # A1 * B_top: 2x2 * 2x3. This is M_223. Rank 11.
        # A2 * B_bot: 2x1 * 1x3. This is M_213. Rank 6.
        
        U1, V1, W1 = get_223() # 2x2x3
        # U1: (4, 11), V1: (6, 11), W1: (6, 11)
        
        U2, V2, W2 = get_schoolbook(2, 1, 3)
        # U2: (2, 6), V2: (3, 6), W2: (6, 6)
        
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R = R1 + R2
        
        U = np.zeros((6, R))
        V = np.zeros((9, R))
        W = np.zeros((6, R))
        
        # Part 1: A1 * B_top
        # A1 is rows 0,1 of A. Cols 0,1 of A.
        # U1 is for 2x2x3.
        # U indices: i*2+j. i in 0,1. j in 0,1.
        # Global U indices: i*3+j. i in 0,1. j in 0,1.
        for i in range(2):
            for j in range(2):
                U[i*3 + j, :R1] = U1[i*2 + j, :R1]
        
        # V1 is for 2x2x3.
        # V indices: j*2+k. j in 0,1. k in 0,1,2.
        # Global V indices: j*3+k. j in 0,1. k in 0,1,2.
        for j in range(2):
            for k in range(3):
                V[j*3 + k, :R1] = V1[j*2 + k, :R1]
        
        # W1 is for 2x2x3.
        # W indices: k*2+i. k in 0,1,2. i in 0,1.
        # Global W indices: k*2+i.
        W[:, :R1] = W1
        
        # Part 2: A2 * B_bot
        # A2 is row 0,1 of A. Col 2 of A.
        # U2 is for 2x1x3.
        # U2 indices: i*1+j. i in 0,1. j=0.
        # Global U indices: i*3+2.
        for i in range(2):
            U[i*3 + 2, R1:] = U2[i*1 + 0, R1:]
        
        # V2 is for 2x1x3.
        # V2 indices: j*3+k. j=0. k in 0,1,2.
        # Global V indices: j*3+k. j=2.
        for k in range(3):
            V[2*3 + k, R1:] = V2[0*3 + k, R1:]
        
        # W2 is for 2x1x3.
        # W2 indices: k*2+i.
        # Global W indices: k*2+i.
        W[:, R1:] = W2
        
        return U, V, W

    # Block Computations
    
    # 1. C11 (2x2) = A11(2x2)B11(2x2) + A12(2x3)B21(3x2)
    # A11B11: M_222 (Rank 7)
    U_a, V_a, W_a = get_strassen_222()
    add_decomposition(U_a, V_a, W_a, 0, 0, 0, 2, 2, 2)
    
    # A12B21: M_232 (Rank 12)
    U_b, V_b, W_b = get_schoolbook(2, 3, 2)
    add_decomposition(U_b, V_b, W_b, 0, 2, 2, 2, 3, 2)
    
    # 2. C12 (2x3) = A11(2x2)B12(2x3) + A12(2x3)B22(3x3)
    # A11B12: M_223 (Rank 11)
    U_c, V_c, W_c = get_223()
    add_decomposition(U_c, V_c, W_c, 0, 0, 2, 2, 2, 3)
    
    # A12B22: M_233 (Rank 17)
    U_d, V_d, W_d = get_233()
    add_decomposition(U_d, V_d, W_d, 0, 2, 2, 2, 3, 3)
    
    # 3. C21 (2x2) = A21(2x2)B11(2x2) + A22(2x3)B21(3x2)
    # A21B11: M_222 (Rank 7)
    U_e, V_e, W_e = get_strassen_222()
    add_decomposition(U_e, V_e, W_e, 2, 0, 0, 2, 2, 2)
    
    # A22B21: M_232 (Rank 12)
    U_f, V_f, W_f = get_schoolbook(2, 3, 2)
    add_decomposition(U_f, V_f, W_f, 2, 2, 2, 2, 3, 2)
    
    # 4. C22 (2x3) = A21(2x2)B12(2x3) + A22(2x3)B22(3x3)
    # A21B12: M_223 (Rank 11)
    U_g, V_g, W_g = get_223()
    add_decomposition(U_g, V_g, W_g, 2, 0, 2, 2, 2, 3)
    
    # A22B22: M_233 (Rank 17)
    U_h, V_h, W_h = get_233()
    add_decomposition(U_h, V_h, W_h, 2, 2, 2, 2, 3, 3)
    
    # Construct final matrices
    if not U_cols:
        # Fallback to schoolbook if something went wrong
        U = np.zeros((nm, nm*mp))
        V = np.zeros((mp, nm*mp))
        W = np.zeros((pn, nm*mp))
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Verify
    # Simple check
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       s = 0
    #       for r in range(U.shape[1]):
    #         s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #       if abs(s-1) > 1e-9:
    #         print(f"Error at {i},{j},{k}: {s}")
    #         return {"U": np.zeros((nm,1)).tolist(), "V": np.zeros((mp,1)).tolist(), "W": np.zeros((pn,1)).tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
