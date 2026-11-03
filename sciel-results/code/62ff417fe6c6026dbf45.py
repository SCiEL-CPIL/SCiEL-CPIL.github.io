"""
Program for matrix multiplication tensor decomposition.
Task: Find a rank-minimal exact tensor decomposition for 2x5 x 5x6 matrix multiplication.
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m x p using a hybrid Strassen/Schoolbook decomposition.
    
    Strategy:
    1. Split the middle dimension m=5 into m1=2 and m2=3.
    2. For the m1=2 part (A is 2x2, B is 2x6), split B into three 2x2 blocks.
       Use Strassen's algorithm (rank 7) for each 2x2x2 multiplication.
       Total rank contribution: 3 * 7 = 21.
    3. For the m2=3 part (A is 2x3, B is 3x6), split B into three 3x2 blocks.
       Use Schoolbook algorithm (rank 12) for each 2x3x2 multiplication.
       Total rank contribution: 3 * 12 = 36.
    4. Total rank = 21 + 36 = 57.
    """
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Pre-allocate lists for U, V, W
    # U: (n*m, rank) -> (10, rank)
    # V: (m*p, rank) -> (30, rank)
    # W: (p*n, rank) -> (12, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Part 1: Strassen for j in {0, 1}
    # A_sub: i in {0,1}, j in {0,1}
    # B_sub: j in {0,1}, k in {2b, 2b+1}
    # C_sub: i in {0,1}, k in {2b, 2b+1}
    
    # Strassen coefficients
    # U (A) coefficients for M1..M7
    U_strassen = [
        [(0, 1), (1, 1)],      # M1: A00 + A11
        [(1, 1), (1, 1)],      # M2: A10 + A11 (Note: A10 is i=1, j=0) -> (1,0) and (1,1)
        [(0, 1)],              # M3: A00
        [(1, 1)],              # M4: A11
        [(0, 1), (0, 1)],      # M5: A00 + A01 -> (0,0) and (0,1)
        [(1, 1), (-1, 1)],     # M6: A10 - A00 -> (1,0) and (0,0) with sign
        [(0, 1), (-1, 1)]      # M7: A01 - A11 -> (0,1) and (1,1) with sign
    ]
    # Correction: U_strassen entries should be (i, j, coeff)
    U_strassen = [
        [(0, 0, 1), (1, 1, 1)],      # M1: A00 + A11
        [(1, 0, 1), (1, 1, 1)],      # M2: A10 + A11
        [(0, 0, 1)],                 # M3: A00
        [(1, 1, 1)],                 # M4: A11
        [(0, 0, 1), (0, 1, 1)],      # M5: A00 + A01
        [(1, 0, 1), (0, 0, -1)],     # M6: A10 - A00
        [(0, 1, 1), (1, 1, -1)]      # M7: A01 - A11
    ]
    
    V_strassen = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    W_strassen = [
        [(0, 0, 1), (1, 0, 1), (0, 1, -1), (1, 1, 1)], # C00 = M1 + M4 - M5 + M7
        [(0, 1, 1), (1, 1, 1)],                        # C01 = M3 + M5
        [(1, 0, 1), (1, 1, 1)],                        # C10 = M2 + M4
        [(0, 0, 1), (1, 0, -1), (0, 1, 1), (1, 1, 1)] # C11 = M1 - M2 + M3 + M6
    ]
    # Note: W index is (k, i). 
    # In W_strassen above, I used (i, k) in my head? 
    # C_{i,k}. W index is k*n + i.
    # Let's re-verify W_strassen indices.
    # C00 (i=0, k=0): M1 + M4 - M5 + M7
    # C01 (i=0, k=1): M3 + M5
    # C10 (i=1, k=0): M2 + M4
    # C11 (i=1, k=1): M1 - M2 + M3 + M6
    
    # W_strassen[m] should contain (k, i, coeff)
    W_strassen = [
        [(0, 0, 1), (0, 1, 1), (1, 0, -1), (1, 1, 1)], # C00 (k=0, i=0)
        [(1, 0, 1), (1, 1, 1)],                        # C01 (k=1, i=0)
        [(0, 1, 1), (0, 1, 1)],                        # C10 (k=0, i=1) -> M2+M4
        [(0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, 1)] # C11 (k=1, i=1) -> M1-M2+M3+M6
    ]
    # Wait, M2 and M4 contribute to C10.
    # C10 = M2 + M4.
    # My W_strassen[2] is for C10.
    # It should have coeffs for M2 and M4.
    # But W_strassen is indexed by m (0..6).
    # The structure is: For each m (0..6), which (k,i) does it contribute to with what coeff?
    # It's better to iterate m and add to W.
    
    for b in range(3):
        k_base = 2 * b
        for m_idx in range(7):
            # Create U, V, W columns
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            # Fill U
            for i, j, coeff in U_strassen[m_idx]:
                u_col[i * m + j] += coeff
                
            # Fill V
            for j, k_local, coeff in V_strassen[m_idx]:
                k_global = k_base + k_local
                v_col[j * p + k_global] += coeff
                
            # Fill W
            # We need to sum contributions to C from this m_idx?
            # No, the tensor decomposition is sum_c U_c V_c W_c = T.
            # Each column c corresponds to one multiplication in Strassen.
            # The output C is a linear combination of these products.
            # So W_c should be the vector representing the contribution of the m-th Strassen product to the final C.
            # C_{i,k} = sum_m coeff_{m, i, k} * (Product_m)
            # So for a fixed m_idx, W_{k, i} = coeff_{m_idx, i, k}.
            
            # Let's define the coeffs for W based on m_idx
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            coeffs = {}
            if m_idx == 0: # M1
                coeffs[(0, 0)] = 1
                coeffs[(1, 1)] = 1
            elif m_idx == 1: # M2
                coeffs[(1, 0)] = 1
                coeffs[(1, 1)] = -1
            elif m_idx == 2: # M3
                coeffs[(0, 1)] = 1
                coeffs[(1, 1)] = 1
            elif m_idx == 3: # M4
                coeffs[(0, 0)] = 1
                coeffs[(1, 0)] = 1
            elif m_idx == 4: # M5
                coeffs[(0, 0)] = -1
                coeffs[(0, 1)] = 1
            elif m_idx == 5: # M6
                coeffs[(1, 1)] = 1
            elif m_idx == 6: # M7
                coeffs[(0, 0)] = 1
            
            for (k_local, i), coeff in coeffs.items():
                k_global = k_base + k_local
                w_col[k_global * n + i] += coeff
                
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    # Part 2: Schoolbook for j in {2, 3, 4}
    for b in range(3):
        k_base = 2 * b
        for j in range(2, 5):
            for i in range(2):
                for k_local in range(2):
                    k_global = k_base + k_local
                    u_col = [0.0] * (n * m)
                    v_col = [0.0] * (m * p)
                    w_col = [0.0] * (p * n)
                    
                    u_col[i * m + j] = 1.0
                    v_col[j * p + k_global] = 1.0
                    w_col[k_global * n + i] = 1.0
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                    
    # Transpose to get rows
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
