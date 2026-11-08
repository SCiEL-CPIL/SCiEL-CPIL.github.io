import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W matrices.
    # U: (n*m, rank) -> (12, rank)
    # V: (m*p, rank) -> (16, rank)
    # W: (p*n, rank) -> (12, rank)
    
    # We use a block decomposition.
    # A is 3x4. Split into A_top (2x4) and A_bot (1x4).
    # B is 4x4. Split into B11, B12, B21, B22 (each 2x2).
    
    # Indices for A (row, col) in U:
    # A_top: rows 0,1. Cols 0,1,2,3.
    # A_bot: row 2. Cols 0,1,2,3.
    
    # Indices for B (row, col) in V:
    # B11: rows 0,1. Cols 0,1.
    # B12: rows 0,1. Cols 2,3.
    # B21: rows 2,3. Cols 0,1.
    # B22: rows 2,3. Cols 2,3.
    
    # Indices for C (row, col) in W:
    # C is 4x3. W[k*n+i, r] -> W[col*3+row, r]?
    # Task says: W[k*n+i, r].
    # Output C is 4x3.
    # C[row, col].
    # W index: k*n+i.
    # Usually C[i, j] corresponds to W[j*n+i, r] if C is n x p?
    # Here A is n x m (3x4), B is m x p (4x4).
    # C is n x p (3x4).
    # Wait, the task says "3 by 4 matrix by 4 by 4 matrix".
    # Output is 3x4.
    # W[k*n+i, r].
    # If C is 3x4, then i is row (0..2), k is col (0..3).
    # W[k*3+i, r].
    
    # Let's verify the W index convention.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # n=3.
    # So W[col*3+row, r].
    
    U_rows = n * m # 12
    V_rows = m * p # 16
    W_rows = p * n # 12
    
    # We will accumulate the contributions to U, V, W.
    # Since we are summing independent decompositions, we can just append columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomposition(u_vec, v_vec, w_vec):
        """
        u_vec: length 12
        v_vec: length 16
        w_vec: length 12
        """
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def strassen_decomp(a_indices, b_indices, c_indices):
        """
        Adds a Strassen decomposition for a 2x2 multiplication.
        a_indices: list of 4 indices into U (for A11, A12, A21, A22)
        b_indices: list of 4 indices into V (for B11, B12, B21, B22)
        c_indices: list of 4 indices into W (for C11, C12, C21, C22)
        """
        # Strassen formulas for C = A * B
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 + M6
        
        # We need to define the linear combinations for U, V, W.
        
        # M1
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[0]: u[idx] += 1 # A11
        for idx in a_indices[3]: u[idx] += 1 # A22
        for idx in b_indices[0]: v[idx] += 1 # B11
        for idx in b_indices[3]: v[idx] += 1 # B22
        # W contribution: C11 += 1, C22 += 1
        w[c_indices[0]] += 1
        w[c_indices[3]] += 1
        add_decomposition(u, v, w)
        
        # M2
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[2]: u[idx] += 1 # A21
        for idx in a_indices[3]: u[idx] += 1 # A22
        for idx in b_indices[0]: v[idx] += 1 # B11
        # W contribution: C21 += 1
        w[c_indices[2]] += 1
        add_decomposition(u, v, w)
        
        # M3
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[0]: u[idx] += 1 # A11
        for idx in b_indices[1]: v[idx] += 1 # B12
        for idx in b_indices[3]: v[idx] -= 1 # B22
        # W contribution: C12 += 1, C22 -= 1
        w[c_indices[1]] += 1
        w[c_indices[3]] -= 1
        add_decomposition(u, v, w)
        
        # M4
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[3]: u[idx] += 1 # A22
        for idx in b_indices[2]: v[idx] += 1 # B21
        for idx in b_indices[0]: v[idx] -= 1 # B11
        # W contribution: C11 -= 1, C21 += 1
        w[c_indices[0]] -= 1
        w[c_indices[2]] += 1
        add_decomposition(u, v, w)
        
        # M5
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[0]: u[idx] += 1 # A11
        for idx in a_indices[1]: u[idx] += 1 # A12
        for idx in b_indices[3]: v[idx] += 1 # B22
        # W contribution: C11 += 1, C12 += 1
        w[c_indices[0]] += 1
        w[c_indices[1]] += 1
        add_decomposition(u, v, w)
        
        # M6
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[2]: u[idx] += 1 # A21
        for idx in a_indices[0]: u[idx] -= 1 # A11
        for idx in b_indices[0]: v[idx] += 1 # B11
        for idx in b_indices[1]: v[idx] += 1 # B12
        # W contribution: C22 += 1
        w[c_indices[3]] += 1
        add_decomposition(u, v, w)
        
        # M7
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[1]: u[idx] += 1 # A12
        for idx in a_indices[3]: u[idx] -= 1 # A22
        for idx in b_indices[2]: v[idx] += 1 # B21
        for idx in b_indices[3]: v[idx] += 1 # B22
        # W contribution: C11 += 1
        w[c_indices[0]] += 1
        add_decomposition(u, v, w)

    def schoolbook_decomp(a_indices, b_indices, c_indices):
        """
        Adds a schoolbook decomposition for a 2x2 multiplication.
        """
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        
        # A11B11 -> C11
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[0]: u[idx] += 1
        for idx in b_indices[0]: v[idx] += 1
        w[c_indices[0]] += 1
        add_decomposition(u, v, w)
        
        # A12B21 -> C11
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[1]: u[idx] += 1
        for idx in b_indices[2]: v[idx] += 1
        w[c_indices[0]] += 1
        add_decomposition(u, v, w)
        
        # A11B12 -> C12
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[0]: u[idx] += 1
        for idx in b_indices[1]: v[idx] += 1
        w[c_indices[1]] += 1
        add_decomposition(u, v, w)
        
        # A12B22 -> C12
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[1]: u[idx] += 1
        for idx in b_indices[3]: v[idx] += 1
        w[c_indices[1]] += 1
        add_decomposition(u, v, w)
        
        # A21B11 -> C21
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[2]: u[idx] += 1
        for idx in b_indices[0]: v[idx] += 1
        w[c_indices[2]] += 1
        add_decomposition(u, v, w)
        
        # A22B21 -> C21
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[3]: u[idx] += 1
        for idx in b_indices[2]: v[idx] += 1
        w[c_indices[2]] += 1
        add_decomposition(u, v, w)
        
        # A21B12 -> C22
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[2]: u[idx] += 1
        for idx in b_indices[1]: v[idx] += 1
        w[c_indices[3]] += 1
        add_decomposition(u, v, w)
        
        # A22B22 -> C22
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[3]: u[idx] += 1
        for idx in b_indices[3]: v[idx] += 1
        w[c_indices[3]] += 1
        add_decomposition(u, v, w)

    # Define indices for A blocks
    # A_top (2x4):
    # A_TL (2x2): rows 0,1. cols 0,1.
    # Indices in U: row*m + col.
    # A_TL: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5.
    a_TL = [[0], [1], [4], [5]] # A11, A12, A21, A22 of the 2x2 block
    # A_TR (2x2): rows 0,1. cols 2,3.
    # A_TR: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7.
    a_TR = [[2], [3], [6], [7]]
    
    # A_bot (1x4):
    # A_BL (1x2): row 2. cols 0,1.
    # A_BL: (2,0)->8, (2,1)->9.
    # This is a 1x2 block. We treat it as 2x2 with zeros?
    # No, schoolbook for 1x2 by 2x2 is rank 4.
    # We will handle bottom separately.
    
    # Define indices for B blocks
    # B11 (2x2): rows 0,1. cols 0,1.
    # Indices in V: row*p + col.
    # B11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5.
    b_11 = [[0], [1], [4], [5]]
    # B12 (2x2): rows 0,1. cols 2,3.
    # B12: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7.
    b_12 = [[2], [3], [6], [7]]
    # B21 (2x2): rows 2,3. cols 0,1.
    # B21: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13.
    b_21 = [[8], [9], [12], [13]]
    # B22 (2x2): rows 2,3. cols 2,3.
    # B22: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15.
    b_22 = [[10], [11], [14], [15]]
    
    # Define indices for C blocks
    # C is 3x4. W index: col*3 + row.
    # C_TL (2x2): rows 0,1. cols 0,1.
    # C_TL: (0,0)->0, (0,1)->3, (1,0)->1, (1,1)->4.
    # Wait, W[k*n+i].
    # C[row, col].
    # C_TL[0,0] -> W[0*3+0] = 0.
    # C_TL[0,1] -> W[1*3+0] = 3.
    # C_TL[1,0] -> W[0*3+1] = 1.
    # C_TL[1,1] -> W[1*3+1] = 4.
    c_TL = [[0], [3], [1], [4]]
    
    # C_TR (2x2): rows 0,1. cols 2,3.
    # C_TR[0,0] (C[0,2]) -> W[2*3+0] = 6.
    # C_TR[0,1] (C[0,3]) -> W[3*3+0] = 9.
    # C_TR[1,0] (C[1,2]) -> W[2*3+1] = 7.
    # C_TR[1,1] (C[1,3]) -> W[3*3+1] = 10.
    c_TR = [[6], [9], [7], [10]]
    
    # C_BL (1x2): row 2. cols 0,1.
    # C_BL[0,0] (C[2,0]) -> W[0*3+2] = 2.
    # C_BL[0,1] (C[2,1]) -> W[1*3+2] = 5.
    # We need to map this to the schoolbook function which expects 4 indices.
    # We can create a dummy 2x2 block for C_BL with zeros in the second row?
    # Or just write a specific schoolbook for 1x2.
    
    def schoolbook_decomp_1x2(a_indices, b_indices, c_indices):
        """
        a_indices: 2 indices (A1, A2)
        b_indices: 4 indices (B11, B12, B21, B22)
        c_indices: 2 indices (C1, C2)
        """
        # C1 = A1*B11 + A2*B21
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[0]: u[idx] += 1
        for idx in b_indices[0]: v[idx] += 1
        w[c_indices[0]] += 1
        add_decomposition(u, v, w)
        
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[1]: u[idx] += 1
        for idx in b_indices[2]: v[idx] += 1
        w[c_indices[0]] += 1
        add_decomposition(u, v, w)
        
        # C2 = A1*B12 + A2*B22
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[0]: u[idx] += 1
        for idx in b_indices[1]: v[idx] += 1
        w[c_indices[1]] += 1
        add_decomposition(u, v, w)
        
        u = [0]*12; v = [0]*16; w = [0]*12
        for idx in a_indices[1]: u[idx] += 1
        for idx in b_indices[3]: v[idx] += 1
        w[c_indices[1]] += 1
        add_decomposition(u, v, w)

    # Top Left: A_TL * B_11 + A_TR * B_21
    # This is sum of two 2x2 multiplications.
    strassen_decomp(a_TL, b_11, c_TL)
    strassen_decomp(a_TR, b_21, c_TL)
    
    # Top Right: A_TL * B_12 + A_TR * B_22
    strassen_decomp(a_TL, b_12, c_TR)
    strassen_decomp(a_TR, b_22, c_TR)
    
    # Bottom Left: A_BL * B_11 + A_BR * B_21
    # A_BL indices: 8, 9.
    a_BL = [[8], [9]]
    # A_BR indices: 10, 11.
    a_BR = [[10], [11]]
    
    # C_BL indices: 2, 5.
    c_BL = [[2], [5]]
    
    schoolbook_decomp_1x2(a_BL, b_11, c_BL)
    schoolbook_decomp_1x2(a_BR, b_21, c_BL)
    
    # Bottom Right: A_BL * B_12 + A_BR * B_22
    # C_BR indices: C[2,2]->W[2*3+2]=8, C[2,3]->W[3*3+2]=11.
    c_BR = [[8], [11]]
    
    schoolbook_decomp_1x2(a_BL, b_12, c_BR)
    schoolbook_decomp_1x2(a_BR, b_22, c_BR)
    
    # Convert lists to arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
