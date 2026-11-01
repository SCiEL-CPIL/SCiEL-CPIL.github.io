import numpy as np

def construct():
    n, m, p = 2, 5, 6
    
    # Helper to create Strassen 2x2x2 factors
    def get_strassen_222():
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
        # m=2, p=2, n=2
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1: (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # c00 += M1
        W[3, 0] = 1  # c11 += M1
        
        # M2: (a10+a11)b00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # c10 += M2
        W[3, 1] = -1 # c11 -= M2
        
        # M3: a00(b10-b11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # c01 += M3
        W[3, 2] = 1  # c11 += M3
        
        # M4: a11(b01-b00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # c00 += M4
        W[2, 3] = 1  # c10 += M4
        
        # M5: (a00+a01)b11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # c00 -= M5
        W[1, 4] = 1  # c01 += M5
        
        # M6: (a10-a00)(b00+b10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1  # c11 += M6
        
        # M7: (a01-a11)(b11+b01)
        U[1, 6] = 1; U[3, 6] = -1
        V[3, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1  # c00 += M7
        W[3, 6] = 1  # c11 += M7
        
        return U, V, W

    U_total = []
    V_total = []
    W_total = []
    
    # Add two M_2,2,6 blocks (using Strassen)
    U_222, V_222, W_222 = get_strassen_222()
    
    # Block 1: A cols 0,1. B rows 0,1. C cols 0,1.
    U_b1 = U_222
    V_b1 = np.zeros((12, 7), dtype=int) # m*p = 5*6 = 30? No, for sub-block m=2, p=2.
    # Wait, the global V has shape (m*p, rank) = (30, rank).
    # The sub-block V has shape (2*2, 7) = (4, 7).
    # We need to embed it into the global V.
    # Global V index is j*p + k.
    # Sub-block j in {0,1}, k in {0,1}.
    # Global indices: 0*6+0=0, 0*6+1=1, 1*6+0=6, 1*6+1=7.
    # So V_b1_global[0, :] = V_222[0, :]
    # V_b1_global[1, :] = V_222[1, :]
    # V_b1_global[6, :] = V_222[2, :]
    # V_b1_global[7, :] = V_222[3, :]
    
    def embed_222_to_226(U_sub, V_sub, W_sub, j_offset, k_offset):
        # U_sub: (4, r) -> Global U: (10, r)
        # Global U index: i*m + j. m=5.
        # Sub-block j in {0,1}. i in {0,1}.
        # Global indices: i*5 + (j_offset + j_sub).
        
        # V_sub: (4, r) -> Global V: (30, r)
        # Global V index: j*p + k. p=6.
        # Sub-block k in {0,1}. j in {0,1} (relative to j_offset? No, j is intermediate).
        # Wait, in M_2,2,6, the intermediate dimension is 2.
        # In the full problem, intermediate is 5.
        # So j_sub maps to j_global = j_offset + j_sub.
        # k_sub maps to k_global = k_offset + k_sub.
        
        # W_sub: (4, r) -> Global W: (12, r)
        # Global W index: k*n + i. n=2.
        # Sub-block k in {0,1}. i in {0,1}.
        # Global indices: (k_offset + k_sub)*2 + i_sub.
        
        r = U_sub.shape[1]
        U_g = np.zeros((10, r), dtype=int)
        V_g = np.zeros((30, r), dtype=int)
        W_g = np.zeros((12, r), dtype=int)
        
        for i in range(2):
            for j in range(2):
                U_g[i*5 + (j_offset + j), :] = U_sub[i*2 + j, :]
                
        for j in range(2):
            for k in range(2):
                V_g[(j_offset + j)*6 + (k_offset + k), :] = V_sub[j*2 + k, :]
                
        for k in range(2):
            for i in range(2):
                W_g[(k_offset + k)*2 + i, :] = W_sub[k*2 + i, :]
                
        return U_g, V_g, W_g

    # Block 1: j_offset=0, k_offset=0
    U1, V1, W1 = embed_222_to_226(U_222, V_222, W_222, 0, 0)
    U_total.append(U1)
    V_total.append(V1)
    W_total.append(W1)
    
    # Block 2: j_offset=2, k_offset=2
    # Note: A uses cols 2,3. B uses rows 2,3. C uses cols 2,3.
    U2, V2, W2 = embed_222_to_226(U_222, V_222, W_222, 2, 2)
    U_total.append(U2)
    V_total.append(V2)
    W_total.append(W2)
    
    # Block 3: M_2,1,6
    # j_offset=4 (col 4 of A). k_offset=4? No, p=6.
    # A col 4. B row 4. C cols 4,5?
    # Wait, M_2,1,6 means m=1, p=6.
    # A is 2x1. B is 1x6.
    # So A uses col 4. B uses row 4. C uses all 6 cols?
    # No, the term is A_4 B_4.
    # A_4 is 2x1. B_4 is 1x6.
    # Product is 2x6.
    # So C uses all 6 columns.
    # So k_offset is not a single offset, it's all 6.
    
    U3 = np.zeros((10, 12), dtype=int)
    V3 = np.zeros((30, 12), dtype=int)
    W3 = np.zeros((12, 12), dtype=int)
    
    for i in range(2):
        for k in range(6):
            t = i*6 + k
            # U: i*m + j. j=4.
            U3[i*5 + 4, t] = 1
            # V: j*p + k. j=4.
            V3[4*6 + k, t] = 1
            # W: k*n + i.
            W3[k*2 + i, t] = 1
            
    U_total.append(U3)
    V_total.append(V3)
    W_total.append(W3)
    
    U_final = np.hstack(U_total)
    V_final = np.hstack(V_total)
    W_final = np.hstack(W_total)
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
