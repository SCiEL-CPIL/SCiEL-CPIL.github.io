import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will construct the decomposition for 4x4x4 (Strassen) and 4x4x5 (Standard)
    # Actually, let's just do 4x4x9 directly with a split.
    # Split B into B1 (4x4) and B2 (4x5).
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x5 using Standard (Rank 80)
    # Total Rank 129.
    
    # Base Strassen 2x2x2 terms
    # U, V, W are 4x7 matrices
    # Indices for 2x2: 0,1,2,3
    # A: 0,1,2,3
    # B: 0,1,2,3
    # C: 0,1,2,3
    
    base_U = np.zeros((4, 7))
    base_V = np.zeros((4, 7))
    base_W = np.zeros((4, 7))
    
    # T1: (A11+A22)(B11+B22)
    base_U[0, 0] = 1; base_U[3, 0] = 1
    base_V[0, 0] = 1; base_V[3, 0] = 1
    base_W[0, 0] = 1; base_W[3, 0] = 1
    
    # T2: (A21+A22)B11
    base_U[2, 1] = 1; base_U[3, 1] = 1
    base_V[0, 1] = 1
    base_W[2, 1] = 1; base_W[3, 1] = -1
    
    # T3: A11(B12-B22)
    base_U[0, 2] = 1
    base_V[1, 2] = 1; base_V[3, 2] = -1
    base_W[1, 2] = 1; base_W[3, 2] = 1
    
    # T4: A22(B21-B11)
    base_U[3, 3] = 1
    base_V[2, 3] = 1; base_V[0, 3] = -1
    base_W[0, 3] = 1; base_W[2, 3] = 1
    
    # T5: (A11+A12)B22
    base_U[0, 4] = 1; base_U[1, 4] = 1
    base_V[3, 4] = 1
    base_W[0, 4] = -1; base_W[1, 4] = 1
    
    # T6: (A21-A11)(B11+B12)
    base_U[2, 5] = 1; base_U[0, 5] = -1
    base_V[0, 5] = 1; base_V[1, 5] = 1
    base_W[3, 5] = 1
    
    # T7: (A12-A22)(B21+B22)
    base_U[1, 6] = 1; base_U[3, 6] = -1
    base_V[2, 6] = 1; base_V[3, 6] = 1
    base_W[0, 6] = 1; base_W[2, 6] = 1
    
    # Now build 4x4x4 Strassen
    # 7 Strassen terms M1..M7
    # Each Mk has A-part, B-part, and contributes to C-parts.
    
    # A-parts:
    # M1: A11+A22
    # M2: A21+A22
    # M3: A11
    # M4: A22
    # M5: A11+A12
    # M6: A21-A11
    # M7: A12-A22
    
    # B-parts:
    # M1: B11+B22
    # M2: B11
    # M3: B12-B22
    # M4: B21-B11
    # M5: B22
    # M6: B11+B12
    # M7: B21+B22
    
    # C-contributions:
    # C11: M1 + M4 - M5 + M7
    # C12: M3 + M5
    # C21: M2 + M4
    # C22: M1 - M2 + M3 + M6
    
    # We will generate 49 terms.
    U_444 = []
    V_444 = []
    W_444 = []
    
    # Helper to get block indices
    # Block (r, c) of 2x2 in 4x4
    # r, c in {0, 1}
    # Rows: r*2, r*2+1
    # Cols: c*2, c*2+1
    def get_block_indices(r, c):
        return [r*2 + dr for dr in [0, 1] for dc in [0, 1] for _ in [0]] # Wait, row-major
        # Actually, index = row * 4 + col
        # Block (r,c) covers rows r*2, r*2+1 and cols c*2, c*2+1
        idx = []
        for dr in range(2):
            for dc in range(2):
                idx.append((r*2 + dr) * 4 + (c*2 + dc))
        return idx

    # Define A-parts as list of (block_row, block_col, coeff)
    A_parts = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],          # M3
        [(1,1,1)],          # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    
    B_parts = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],          # M2
        [(0,1,1), (1,1,-1)],# M3
        [(1,0,1), (0,0,-1)],# M4
        [(1,1,1)],          # M5
        [(0,0,1), (0,1,1)], # M6
        [(1,0,1), (1,1,1)]  # M7
    ]
    
    # C-parts: for each C block, list of (Mk, coeff)
    # C11: M1(1), M4(1), M5(-1), M7(1)
    # C12: M3(1), M5(1)
    # C21: M2(1), M4(1)
    # C22: M1(1), M2(-1), M3(1), M6(1)
    
    C_parts = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)]
    }
    
    # For each Mk (0..6) and each base term Tj (0..6)
    for k in range(7):
        for j in range(7):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 16
            w_vec = [0.0] * 16
            
            # Fill U
            for (br, bc, coeff) in A_parts[k]:
                idxs = get_block_indices(br, bc)
                for idx_local, idx_global in enumerate(idxs):
                    u_vec[idx_global] += base_U[idx_local, j] * coeff
            
            # Fill V
            for (br, bc, coeff) in B_parts[k]:
                idxs = get_block_indices(br, bc)
                for idx_local, idx_global in enumerate(idxs):
                    v_vec[idx_global] += base_V[idx_local, j] * coeff
            
            # Fill W
            # We need to add the contribution of Mk to the C blocks.
            # The base W vector for Tj gives the coefficients for the 2x2 C block.
            # We need to map these to the global C indices.
            # But which C block?
            # Mk contributes to multiple C blocks with different coefficients.
            # So for each C block that Mk contributes to, we add the base W vector scaled by the coefficient.
            
            # We need to know which C blocks Mk contributes to.
            # From C_parts, we can invert: for each Mk, find C blocks.
            # Or just iterate over all C blocks.
            for (cr, cc) in [(0,0), (0,1), (1,0), (1,1)]:
                # Check if Mk is in C_parts[(cr, cc)]
                coeff = 0
                for (mk, c_coeff) in C_parts[(cr, cc)]:
                    if mk == k:
                        coeff = c_coeff
                        break
                if coeff != 0:
                    idxs = get_block_indices(cr, cc)
                    for idx_local, idx_global in enumerate(idxs):
                        w_vec[idx_global] += base_W[idx_local, j] * coeff
            
            U_444.append(u_vec)
            V_444.append(v_vec)
            W_444.append(w_vec)
            
    # Now add 4x4x5 standard decomposition
    # Rank 80
    U_445 = []
    V_445 = []
    W_445 = []
    for i in range(4):
        for j in range(4):
            for k in range(5):
                u_vec = [0.0] * 16
                v_vec = [0.0] * 20 # 4x5
                w_vec = [0.0] * 20 # 5x4? No, W is p*n = 5*4=20
                # U index: i*4+j
                u_vec[i*4+j] = 1
                # V index: j*5+k
                v_vec[j*5+k] = 1
                # W index: k*4+i
                w_vec[k*4+i] = 1
                U_445.append(u_vec)
                V_445.append(v_vec)
                W_445.append(w_vec)
                
    # Combine
    # U: 16 rows for A, 20 rows for B? No.
    # U is (n*m) x rank. n=4, m=4. 16 rows.
    # V is (m*p) x rank. m=4, p=9. 36 rows.
    # W is (p*n) x rank. p=9, n=4. 36 rows.
    
    # My U_444 has 16 rows. Correct.
    # My V_444 has 16 rows? No, B is 4x4. 16 rows.
    # But total B is 4x9. 36 rows.
    # So for V_444, I need to pad to 36 rows.
    # The first 16 rows correspond to B1 (4x4).
    # The last 20 rows are 0.
    
    # For W_444, I have 16 rows? No, C is 4x4. 16 rows.
    # But total C is 4x9. 36 rows.
    # W index is k*n+i.
    # For B1 (cols 0-3), k is 0-3.
    # So W_444 corresponds to k=0..3.
    # I need to pad to 36 rows.
    
    U_total = []
    V_total = []
    W_total = []
    
    for i in range(49):
        u = U_444[i]
        v = [0.0] * 36
        for j in range(16):
            v[j] = V_444[i][j]
        w = [0.0] * 36
        for j in range(16):
            w[j] = W_444[i][j]
        U_total.append(u)
        V_total.append(v)
        W_total.append(w)
        
    for i in range(80):
        u = U_445[i]
        v = [0.0] * 36
        for j in range(20):
            v[16+j] = V_445[i][j]
        w = [0.0] * 36
        for j in range(20):
            w[16+j] = W_445[i][j]
        U_total.append(u)
        V_total.append(v)
        W_total.append(w)
        
    return {"U": U_total, "V": V_total, "W": W_total}
