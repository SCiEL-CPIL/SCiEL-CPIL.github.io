import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the factors U, V, W by appending columns.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_val, v_val, w_val, i_idx, j_idx, k_idx):
        # i_idx: index in U (0 to n*m-1)
        # j_idx: index in V (0 to m*p-1)
        # k_idx: index in W (0 to p*n-1)
        U_cols.append([0.0] * (n * m))
        V_cols.append([0.0] * (m * p))
        W_cols.append([0.0] * (p * n))
        U_cols[-1][i_idx] = u_val
        V_cols[-1][j_idx] = v_val
        W_cols[-1][k_idx] = w_val

    def schoolbook(n_block, m_block, p_block, i_off, j_off, k_off):
        # i_off: offset in i (0..n-1)
        # j_off: offset in j (0..m-1)
        # k_off: offset in k (0..p-1)
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    add_term(1, 1, 1, u_idx, v_idx, w_idx)

    def strassen(n_block, m_block, p_block, i_off, j_off, k_off):
        # Only valid if n_block==m_block==p_block==2
        if n_block != 2 or m_block != 2 or p_block != 2:
            schoolbook(n_block, m_block, p_block, i_off, j_off, k_off)
            return
            
        # Strassen for 2x2x2
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to the global indices.
        # A indices:
        # A11: i=i_off, j=j_off
        # A12: i=i_off, j=j_off+1
        # A21: i=i_off+1, j=j_off
        # A22: i=i_off+1, j=j_off+1
        
        # B indices:
        # B11: j=j_off, k=k_off
        # B12: j=j_off, k=k_off+1
        # B21: j=j_off+1, k=k_off
        # B22: j=j_off+1, k=k_off+1
        
        # C indices:
        # C11: i=i_off, k=k_off
        # C12: i=i_off, k=k_off+1
        # C21: i=i_off+1, k=k_off
        # C22: i=i_off+1, k=k_off+1
        
        def add_strassen_term(coeffs_A, coeffs_B, coeffs_C):
            # coeffs_A: dict { (i_rel, j_rel): coeff }
            # coeffs_B: dict { (j_rel, k_rel): coeff }
            # coeffs_C: dict { (i_rel, k_rel): coeff }
            # This adds a single rank-1 term? No, Strassen terms are sums of products.
            # Each M is a product of two linear combinations.
            # This corresponds to one rank-1 term in the tensor decomposition?
            # No, $M = (A \cdot u) (B \cdot v)$.
            # This is a rank-1 term in the bilinear form?
            # Yes, $C_{ij} = M \cdot w_{ij}$.
            # So we add one term to U, V, W.
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for (i_r, j_r), c in coeffs_A.items():
                idx = (i_off + i_r) * m + (j_off + j_r)
                u_vec[idx] = c
                
            for (j_r, k_r), c in coeffs_B.items():
                idx = (j_off + j_r) * p + (k_off + k_r)
                v_vec[idx] = c
                
            for (i_r, k_r), c in coeffs_C.items():
                idx = (k_off + k_r) * n + (i_off + i_r)
                w_vec[idx] = c
                
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

        # M1
        add_strassen_term(
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): -1, (0,1): 0, (1,0): 0} # Placeholder, C coeffs handled below
        )
        # Actually, we need to add 7 terms, each contributing to C with specific coefficients.
        # It is easier to just add the 7 terms and let the W vector encode the C coefficients.
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        add_strassen_term(
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): -1, (1,0): 0, (0,1): 0} # C11+=1, C22-=1
        )
        # Wait, C11 = M1 + M4 - M5 + M7. C22 = M1 - M2 + M3 + M6.
        # So M1 contributes +1 to C11 and +1 to C22.
        add_strassen_term(
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}
        )
        
        # Term 2: M2 = (A21+A22)B11
        add_strassen_term(
            {(1,0): 1, (1,1): 1}, 
            {(0,0): 1}, 
            {(1,0): 1, (1,1): -1} # C21+=1, C22-=1
        )
        
        # Term 3: M3 = A11(B12-B22)
        add_strassen_term(
            {(0,0): 1}, 
            {(0,1): 1, (1,1): -1}, 
            {(0,1): 1, (1,1): 1} # C12+=1, C22+=1
        )
        
        # Term 4: M4 = A22(B21-B11)
        add_strassen_term(
            {(1,1): 1}, 
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1, (1,0): 1} # C11+=1, C21+=1
        )
        
        # Term 5: M5 = (A11+A12)B22
        add_strassen_term(
            {(0,0): 1, (0,1): 1}, 
            {(1,1): 1}, 
            {(0,0): -1, (0,1): 1} # C11-=1, C12+=1
        )
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        add_strassen_term(
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1, (0,1): 1}, 
            {(1,1): 1} # C22+=1
        )
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        add_strassen_term(
            {(0,1): 1, (1,1): -1}, 
            {(1,0): 1, (1,1): 1}, 
            {(0,0): 1} # C11+=1
        )

    # Decompose 3x3x7
    # Split m=3 into 2+1
    # A: 3x3 -> [2x2, 2x1; 1x2, 1x1]
    # B: 3x7 -> [2x7; 1x7]
    # C: 3x7 -> [2x7; 1x7]
    
    # Block 1: A(0:2, 0:2) * B(0:2, 0:7) -> C(0:2, 0:7)
    # A is 2x2, B is 2x7.
    # Split B into 2x2, 2x2, 2x2, 2x1.
    # C(0:2, 0:2): Strassen
    strassen(2, 2, 2, 0, 0, 0)
    # C(0:2, 2:4): Strassen
    strassen(2, 2, 2, 0, 0, 2)
    # C(0:2, 4:6): Strassen
    strassen(2, 2, 2, 0, 0, 4)
    # C(0:2, 6:7): Schoolbook (2x2x1)
    schoolbook(2, 2, 1, 0, 0, 6)
    
    # Block 2: A(0:2, 2:3) * B(2:3, 0:7) -> C(0:2, 0:7)
    # A is 2x1, B is 1x7.
    # C is 2x7.
    # This is 2x1x7.
    # Split B into 1x2, 1x2, 1x2, 1x1.
    # C(0:2, 0:2): 2x1x2
    schoolbook(2, 1, 2, 0, 2, 0)
    # C(0:2, 2:4): 2x1x2
    schoolbook(2, 1, 2, 0, 2, 2)
    # C(0:2, 4:6): 2x1x2
    schoolbook(2, 1, 2, 0, 2, 4)
    # C(0:2, 6:7): 2x1x1
    schoolbook(2, 1, 1, 0, 2, 6)
    
    # Block 3: A(2:3, 0:2) * B(0:2, 0:7) -> C(2:3, 0:7)
    # A is 1x2, B is 2x7.
    # C is 1x7.
    # Split B into 2x2, 2x2, 2x2, 2x1.
    # C(2:3, 0:2): 1x2x2
    schoolbook(1, 2, 2, 2, 0, 0)
    # C(2:3, 2:4): 1x2x2
    schoolbook(1, 2, 2, 2, 0, 2)
    # C(2:3, 4:6): 1x2x2
    schoolbook(1, 2, 2, 2, 0, 4)
    # C(2:3, 6:7): 1x2x1
    schoolbook(1, 2, 1, 2, 0, 6)
    
    # Block 4: A(2:3, 2:3) * B(2:3, 0:7) -> C(2:3, 0:7)
    # A is 1x1, B is 1x7.
    # C is 1x7.
    # This is 1x1x7.
    schoolbook(1, 1, 7, 2, 2, 0)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
