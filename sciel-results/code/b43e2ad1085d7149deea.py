"""Optimized tensor decomposition for 2x4 * 4x4 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # We will build a decomposition using Strassen's algorithm on the 2x2 block structure.
    # A is 2x4, B is 4x4.
    # Let A = [A_L | A_R] where A_L, A_R are 2x2.
    # Let B = [[B_11, B_12], [B_21, B_22]] where each block is 2x2.
    # C = A * B = [A_L B_11 + A_R B_21 | A_L B_12 + A_R B_22]
    # Let C_L = A_L B_11 + A_R B_21 and C_R = A_L B_12 + A_R B_22.
    
    # We compute C_L and C_R using Strassen-like optimizations.
    # Since A_L and A_R are 2x2, and the B blocks are 2x2, we can use standard 2x2 multiplication.
    # However, the "Strassen" part usually applies to 2x2 matrix multiplication itself.
    # Standard 2x2 mult has rank 8. Strassen 2x2 mult has rank 7.
    # If we use Strassen for the 2x2 multiplications, we can reduce the total rank.
    
    # Let's define the 7 products for Strassen's algorithm for 2x2 matrices X and Y.
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22)Y11
    # M3 = X11(Y21 - Y22)
    # M4 = X22(Y11 - Y10)  -- Wait, standard Strassen is:
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y10 - Y11)
    # M4 = X11(Y01 - Y00)
    # M5 = (X00 + X10)Y11
    # M6 = (X01 - X00)(Y00 + Y10)
    # M7 = (X11 - X01)(Y01 + Y11)
    
    # This reduces 8 multiplications to 7.
    
    # We need to compute two 2x2 matrix products: C_L and C_R.
    # C_L = A_L B_11 + A_R B_21. This is a sum of two 2x2 products.
    # C_R = A_L B_12 + A_R B_22. This is a sum of two 2x2 products.
    
    # If we use Strassen for each of the 4 sub-products (A_L B_11, A_R B_21, A_L B_12, A_R B_22),
    # that would be 4 * 7 = 28 multiplications.
    # But we can potentially share some computations?
    # A_L and A_R are disjoint sets of inputs. B_11, B_21, B_12, B_22 are disjoint sets of inputs.
    # So the inputs to the four Strassen decompositions are all distinct.
    # Therefore, we can just concatenate the 4 Strassen decompositions.
    # Total rank = 28.
    
    # Let's implement this.
    
    # Helper to get Strassen coefficients for 2x2 multiplication
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (i,j) to coeff for A block
    # v_coeffs: dict mapping (k,l) to coeff for B block
    # w_coeffs: dict mapping (i,l) to coeff for C block
    
    def strassen_2x2(A_block, B_block, C_block):
        # A_block is a 2x2 matrix of input variables (symbolic)
        # We represent A_block as a list of 4 variables: [A00, A01, A10, A11]
        # Similarly for B and C.
        # We return 7 terms. Each term is a tuple (u, v, w)
        # u: dict {var_index: coeff}
        # v: dict {var_index: coeff}
        # w: dict {var_index: coeff}
        
        terms = []
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = {0: 1, 3: 1}
        v1 = {0: 1, 3: 1}
        # C00 = M1 - M4 - M5 + M7
        # C01 = M5 + M6
        # C10 = M2 + M3
        # C11 = M1 - M2 - M6 + M7
        
        # Let's define the 7 M's
        # M1
        terms.append((
            {0: 1, 3: 1}, 
            {0: 1, 3: 1}, 
            None # w depends on final combination
        ))
        
        # Actually, it's better to define the 7 multiplications and then sum them into the 4 outputs.
        # Let's define the 7 bilinear forms.
        
        # M1 = (A00 + A11)(B00 + B11)
        m1_u = {0: 1, 3: 1}
        m1_v = {0: 1, 3: 1}
        
        # M2 = (A10 + A11)B00
        m2_u = {2: 1, 3: 1}
        m2_v = {0: 1}
        
        # M3 = A00(B10 - B11)
        m3_u = {0: 1}
        m3_v = {2: 1, 3: -1}
        
        # M4 = A11(B01 - B00)
        m4_u = {3: 1}
        m4_v = {1: 1, 0: -1}
        
        # M5 = (A00 + A10)B11
        m5_u = {0: 1, 2: 1}
        m5_v = {3: 1}
        
        # M6 = (A01 - A00)(B00 + B10)
        m6_u = {1: 1, 0: -1}
        m6_v = {0: 1, 2: 1}
        
        # M7 = (A11 - A01)(B01 + B11)
        m7_u = {3: 1, 1: -1}
        m7_v = {1: 1, 3: 1}
        
        # Now map to C components
        # C00 = M1 - M4 - M5 + M7
        # C01 = M5 + M6
        # C10 = M2 + M3
        # C11 = M1 - M2 - M6 + M7
        
        # We will create 7 tensor terms.
        # For each M, we add it to the relevant C components.
        
        # This returns a list of (u_dict, v_dict, w_dict)
        # But wait, the U, V, W matrices in the final output are global.
        # The A_block indices are local to the 2x2 block.
        # We need to map local indices to global indices.
        
        pass

    # Let's build the global U, V, W directly.
    # A is 2x4. Indices: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]
    # A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
    # Local indices for A_L: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Global: 0,1,4,5
    # Local indices for A_R: 0:(0,2), 1:(0,3), 2:(1,2), 3:(1,3) -> Global: 2,3,6,7
    
    # B is 4x4. Indices: B[0,0]...B[3,3]
    # B_11 = B[0:2, 0:2], B_12 = B[0:2, 2:4]
    # B_21 = B[2:4, 0:2], B_22 = B[2:4, 2:4]
    # Local indices for B_11: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Global: 0,1,4,5
    # Local indices for B_12: 0:(0,2), 1:(0,3), 2:(1,2), 3:(1,3) -> Global: 2,3,6,7
    # Local indices for B_21: 0:(2,0), 1:(2,1), 2:(3,0), 3:(3,1) -> Global: 8,9,12,13
    # Local indices for B_22: 0:(2,2), 1:(2,3), 2:(3,2), 3:(3,3) -> Global: 10,11,14,15
    
    # C is 2x4. Indices: C[0,0]...C[1,3]
    # C_L = C[0:2, 0:2], C_R = C[0:2, 2:4]
    # Local indices for C_L: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Global: 0,1,2,3
    # Local indices for C_R: 0:(0,2), 1:(0,3), 2:(1,2), 3:(1,3) -> Global: 4,5,6,7
    
    # W index convention: W[k*n + i, r].
    # Global C index for (i,k) is k*n + i? No, the task says W[k*n+i, r].
    # So W row index is k*n + i.
    # For C_L (k=0,1; i=0,1):
    # C[0,0] -> i=0, k=0 -> W[0]
    # C[0,1] -> i=0, k=1 -> W[2]
    # C[1,0] -> i=1, k=0 -> W[1]
    # C[1,1] -> i=1, k=1 -> W[3]
    # For C_R (k=2,3; i=0,1):
    # C[0,2] -> i=0, k=2 -> W[4]
    # C[0,3] -> i=0, k=3 -> W[6]
    # C[1,2] -> i=1, k=2 -> W[5]
    # C[1,3] -> i=1, k=3 -> W[7]
    
    # So Global C Index to W Row:
    # 0 -> 0
    # 1 -> 2
    # 2 -> 1
    # 3 -> 3
    # 4 -> 4
    # 5 -> 6
    # 6 -> 5
    # 7 -> 7
    
    # Let's define the mappings.
    # A Global Index: i*m + j. i in {0,1}, j in {0,1,2,3}.
    # A_L local 0 (0,0) -> Global 0
    # A_L local 1 (0,1) -> Global 1
    # A_L local 2 (1,0) -> Global 4
    # A_L local 3 (1,1) -> Global 5
    # A_R local 0 (0,2) -> Global 2
    # A_R local 1 (0,3) -> Global 3
    # A_R local 2 (1,2) -> Global 6
    # A_R local 3 (1,3) -> Global 7
    
    # B Global Index: j*p + k. j in {0,1,2,3}, k in {0,1,2,3}.
    # B_11 local 0 (0,0) -> Global 0
    # B_11 local 1 (0,1) -> Global 1
    # B_11 local 2 (1,0) -> Global 4
    # B_11 local 3 (1,1) -> Global 5
    # B_12 local 0 (0,2) -> Global 2
    # B_12 local 1 (0,3) -> Global 3
    # B_12 local 2 (1,2) -> Global 6
    # B_12 local 3 (1,3) -> Global 7
    # B_21 local 0 (2,0) -> Global 8
    # B_21 local 1 (2,1) -> Global 9
    # B_21 local 2 (3,0) -> Global 12
    # B_21 local 3 (3,1) -> Global 13
    # B_22 local 0 (2,2) -> Global 10
    # B_22 local 1 (2,3) -> Global 11
    # B_22 local 2 (3,2) -> Global 14
    # B_22 local 3 (3,3) -> Global 15
    
    # C Global Index to W Row:
    # C_L local 0 (0,0) -> C[0,0] -> W[0]
    # C_L local 1 (0,1) -> C[0,1] -> W[2]
    # C_L local 2 (1,0) -> C[1,0] -> W[1]
    # C_L local 3 (1,1) -> C[1,1] -> W[3]
    # C_R local 0 (0,2) -> C[0,2] -> W[4]
    # C_R local 1 (0,3) -> C[0,3] -> W[6]
    # C_R local 2 (1,2) -> C[1,2] -> W[5]
    # C_R local 3 (1,3) -> C[1,3] -> W[7]
    
    a_map_L = [0, 1, 4, 5]
    a_map_R = [2, 3, 6, 7]
    b_map_11 = [0, 1, 4, 5]
    b_map_12 = [2, 3, 6, 7]
    b_map_21 = [8, 9, 12, 13]
    b_map_22 = [10, 11, 14, 15]
    c_map_L = [0, 2, 1, 3] # Maps local C_L index to W row index
    c_map_R = [4, 6, 5, 7] # Maps local C_R index to W row index
    
    # Define Strassen terms for a single 2x2 multiplication
    # Returns list of (u_dict, v_dict, w_dict)
    # u_dict keys are local A indices {0,1,2,3}
    # v_dict keys are local B indices {0,1,2,3}
    # w_dict keys are local C indices {0,1,2,3}
    
    strassen_terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    strassen_terms.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})) # C00 += M1, C11 += M1
    # Wait, the w_dict should reflect the contribution to C.
    # C00 = M1 - M4 - M5 + M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 - M6 + M7
    
    # It is easier to define the 7 M's and then accumulate into C.
    # Let's create a list of 7 (u, v) pairs and a matrix W_local (4x7) that maps M's to C's.
    
    M_u = [
        {0:1, 3:1}, # M1
        {2:1, 3:1}, # M2
        {0:1},      # M3
        {3:1},      # M4
        {0:1, 2:1}, # M5
        {1:1, 0:-1},# M6
        {3:1, 1:-1} # M7
    ]
    M_v = [
        {0:1, 3:1}, # M1
        {0:1},      # M2
        {2:1, 3:-1},# M3
        {1:1, 0:-1},# M4
        {3:1},      # M5
        {0:1, 2:1}, # M6
        {1:1, 3:1}  # M7
    ]
    # W_local[c][m] is the coefficient of M_m in C_c
    W_local = np.array([
        [1, 0, 0, -1, -1, 0, 1],  # C00
        [0, 0, 0, 0, 1, 1, 0],    # C01
        [0, 1, 1, 0, 0, 0, 0],    # C10
        [1, -1, 0, 0, 0, -1, 1]   # C11
    ])
    
    # Now, for each of the 4 products, we add 7 terms.
    # Total 28 terms.
    
    U = np.zeros((n*m, 28))
    V = np.zeros((m*p, 28))
    W = np.zeros((p*n, 28))
    
    col = 0
    for product_idx in range(4):
        if product_idx == 0: # A_L B_11 -> C_L
            a_map = a_map_L
            b_map = b_map_11
            c_map = c_map_L
        elif product_idx == 1: # A_R B_21 -> C_L
            a_map = a_map_R
            b_map = b_map_21
            c_map = c_map_L
        elif product_idx == 2: # A_L B_12 -> C_R
            a_map = a_map_L
            b_map = b_map_12
            c_map = c_map_R
        else: # A_R B_22 -> C_R
            a_map = a_map_R
            b_map = b_map_22
            c_map = c_map_R
            
        for m_idx in range(7):
            u_dict = M_u[m_idx]
            v_dict = M_v[m_idx]
            
            for loc_a, coeff in u_dict.items():
                U[a_map[loc_a], col] += coeff
                
            for loc_b, coeff in v_dict.items():
                V[b_map[loc_b], col] += coeff
                
            for loc_c in range(4):
                w_coeff = W_local[loc_c, m_idx]
                if w_coeff != 0:
                    W[c_map[loc_c], col] += w_coeff
                    
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
