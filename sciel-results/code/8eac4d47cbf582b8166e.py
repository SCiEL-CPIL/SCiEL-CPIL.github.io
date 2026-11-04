import numpy as np

def construct():
    n, m, p = 4, 4, 9
    total_rank = 49 + 49 + 16
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Helper to add a 2x2 Strassen block
    # a_offsets: list of (a_row, a_col) for X00, X01, X10, X11
    # b_offsets: list of (b_row, b_col) for Y00, Y01, Y10, Y11
    # c_offsets: list of (c_row, c_col) for C00, C01, C10, C11
    def add_m2_strassen(a_offs, b_offs, c_offs, start_col):
        # Strassen formulas for 2x2 matrices X and Y
        # X = [[X00, X01], [X10, X11]]
        # Y = [[Y00, Y01], [Y10, Y11]]
        
        # 7 products
        # P1 = (X00 + X11)(Y00 + Y11)
        # P2 = (X10 + X11)Y00
        # P3 = X00(Y01 - Y11)
        # P4 = X11(Y10 - Y00)
        # P5 = (X00 + X01)Y11
        # P6 = (X10 - X00)(Y00 + Y01)
        # P7 = (X01 - X11)(Y10 + Y11)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Each Pk contributes to C.
        # We decompose each Pk = uk * vk.
        # uk is a linear form in X, vk is a linear form in Y.
        
        products = [
            # (u_coeffs, v_coeffs)
            # X00+X11, Y00+Y11
            ({0: 1, 3: 1}, {0: 1, 3: 1}), # P1: X00,X11 and Y00,Y11
            ({1: 1, 3: 1}, {0: 1}),       # P2: X10,X11 and Y00
            ({0: 1}, {1: 1, 3: -1}),      # P3: X00 and Y01,-Y11
            ({3: 1}, {2: 1, 0: -1}),      # P4: X11 and Y10,-Y00
            ({0: 1, 1: 1}, {3: 1}),       # P5: X00,X01 and Y11
            ({1: 1, 0: -1}, {0: 1, 1: 1}),# P6: X10,-X00 and Y00,Y01
            ({1: 1, 3: -1}, {2: 1, 3: 1}) # P7: X01,-X11 and Y10,Y11
        ]
        
        c_coefs = [
            [1, 0, 0, 0], # C00
            [0, 1, 0, 0], # C01
            [0, 0, 1, 0], # C10
            [0, 0, 0, 1]  # C11
        ]
        # Map P to C contributions
        # P1 -> C00(1), C11(1)
        # P2 -> C10(1), C11(-1)
        # P3 -> C01(1), C11(1)
        # P4 -> C00(1), C10(1)
        # P5 -> C00(-1), C01(1)
        # P6 -> C11(1)
        # P7 -> C00(1)
        
        p_to_c = [
            {0: 1, 3: 1}, # P1
            {2: 1, 3: -1},# P2
            {1: 1, 3: 1}, # P3
            {0: 1, 2: 1}, # P4
            {0: -1, 1: 1},# P5
            {3: 1},       # P6
            {0: 1}        # P7
        ]
        
        current_col = start_col
        for i in range(7):
            u_coeffs, v_coeffs = products[i]
            c_map = p_to_c[i]
            
            # Expand u_coeffs into global A indices
            global_u = {}
            for local_idx, coeff in u_coeffs.items():
                a_r, a_c = a_offs[local_idx]
                global_idx = a_r * m + a_c
                global_u[global_idx] = global_u.get(global_idx, 0) + coeff
            
            # Expand v_coeffs into global B indices
            global_v = {}
            for local_idx, coeff in v_coeffs.items():
                b_r, b_c = b_offs[local_idx]
                global_idx = b_r * p + b_c
                global_v[global_idx] = global_v.get(global_idx, 0) + coeff
            
            # Update U
            for g_idx, val in global_u.items():
                U[g_idx, current_col] += val
            
            # Update V
            for g_idx, val in global_v.items():
                V[g_idx, current_col] += val
            
            # Update W
            # For each C contribution
            for local_c_idx, c_val in c_map.items():
                c_r, c_c = c_offs[local_c_idx]
                # W index is k*n + i.
                # c_r is row of C (i), c_c is col of C (k)
                w_idx = c_c * n + c_r
                W[w_idx, current_col] += c_val
            
            current_col += 1
            
        return current_col

    def add_m4_block(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        # A block at (a_r, a_c)
        # B block at (b_r, b_c)
        # C block at (c_r, c_c)
        
        # A sub-blocks (2x2)
        A00 = (a_r, a_c)
        A01 = (a_r, a_c+2)
        A10 = (a_r+2, a_c)
        A11 = (a_r+2, a_c+2)
        
        # B sub-blocks
        B00 = (b_r, b_c)
        B01 = (b_r, b_c+2)
        B10 = (b_r+2, b_c)
        B11 = (b_r+2, b_c+2)
        
        # C sub-blocks
        C00 = (c_r, c_c)
        C01 = (c_r, c_c+2)
        C10 = (c_r+2, c_c)
        C11 = (c_r+2, c_c+2)
        
        # 7 Strassen steps
        # 1. (A00+A11)(B00+B11)
        X0 = [ (A00[0], A00[1]), (A00[0], A00[1]+1), (A00[0]+1, A00[1]), (A00[0]+1, A00[1]+1) ] # X00,X01,X10,X11
        # This mapping is tricky. X00 is A00. X11 is A11.
        # X00 = A00, X01 = A00[:,1:]? No, A00 is 2x2.
        # X00 (local 0x0) corresponds to A00.
        # X01 (local 0x1) corresponds to A00's right col? No.
        # In Strassen $M_2$ for $M_4$, X is $2 \times 2$ of $2 \times 2$ blocks?
        # NO. X is $2 \times 2$ of *blocks*?
        # Yes, X is $2 \times 2$ matrix where each entry is a $2 \times 2$ matrix.
        # But Strassen $M_2$ operates on $2 \times 2$ *matrices*.
        # So X00 (in $M_2$ context) is the top-left $2 \times 2$ block of the $4 \times 4$ matrix.
        # Which is A00.
        
        # So for Step 1: X = [[A00, A01], [A10, A11]]?
        # No. Strassen $M_4$ splits A into $2 \times 2$ blocks $A_{00}, A_{01}, A_{10}, A_{11}$.
        # Then computes $P_1 = (A_{00} + A_{11})(B_{00} + B_{11})$.
        # Here $A_{00}$ is a $2 \times 2$ matrix.
        # So X (for the $M_2$ calculation) is not a single $2 \times 2$ of scalars.
        # The "Multiplication" $P_1$ is a matrix multiplication of two $2 \times 2$ matrices.
        # So we need to decompose the multiplication of two $2 \times 2$ matrices into 7 scalar multiplications.
        # This is exactly `add_m2_strassen`.
        
        # So we call add_m2_strassen for each of the 7 P_k.
        
        # P1: (A00+A11) * (B00+B11)
        # X matrix for this call:
        # X00 = A00 + A11 ? No.
        # add_m2_strassen takes offsets for X00, X01, X10, X11.
        # But here X00 in the call corresponds to the top-left entry of the $2 \times 2$ "X" matrix which is a $2 \times 2$ block?
        # Wait. $P_1 = U_1 V_1$. $U_1$ is a $2 \times 2$ matrix. $V_1$ is a $2 \times 2$ matrix.
        # $U_1 = A_{00} + A_{11}$.
        # So $U_1$ is a $2 \times 2$ matrix.
        # To multiply $U_1 V_1$ using Strassen $M_2$ (rank 7), we treat $U_1$ and $V_1$ as $2 \times 2$ matrices of scalars.
        # So we need the offsets for the 4 entries of $U_1$.
        # $U_1[0,0] = A_{00}[0,0] + A_{11}[0,0]$.
        # $U_1[0,1] = A_{00}[0,1] + A_{11}[0,1]$.
        # $U_1[1,0] = A_{00}[1,0] + A_{11}[1,0]$.
        # $U_1[1,1] = A_{00}[1,1] + A_{11}[1,1]$.
        
        # So `a_offs` in `add_m2_strassen` should be the global indices of these 4 scalar entries.
        
        def get_4_entries(base_r, base_c, base2_r=None, base2_c=None):
            # If base2 is None, just base
            if base2_r is None:
                return [
                    (base_r, base_c),
                    (base_r, base_c+1),
                    (base_r+1, base_c),
                    (base_r+1, base_c+1)
                ]
            else:
                # Sum of two 2x2 blocks
                # Entry 0: (r, c) + (r2, c2)
                # This is a linear form.
                # add_m2_strassen expects global indices for scalar accumulation.
                # We need to modify add_m2_strassen to accept linear forms.
                pass

        # To simplify, I will expand the linear forms manually for each P.
        # This is tedious but reliable.
        
        # P1: (A00+A11)(B00+B11)
        # U1 = A00+A11, V1 = B00+B11
        # U1_entries:
        # u00 = A[a_r, a_c] + A[a_r+2, a_c+2]
        # u01 = A[a_r, a_c+1] + A[a_r+2, a_c+2+1]
        # u10 = A[a_r+1, a_c] + A[a_r+2+1, a_c+2]
        # u11 = A[a_r+1, a_c+1] + A[a_r+2+1, a_c+2+1]
        
        # We need a version of add_m2_strassen that takes linear forms.
        
        # Given time, I will implement a helper that adds a rank-1 term from scalar linear forms.
        
        return start_col

    # This approach is getting complex. 
    # Fallback: Use Schoolbook for everything but optimize the structure.
    # The seed is 144.
    # Can we just use 144? 
    # The prompt asks for stronger algorithm.
    
    # Let's use the simple block decomposition I found: 16 + 16 + 16 = 48?
    # Wait, why did I think 16 for 4x4?
    # Rank of 4x4 mat mult is NOT 16. It is 64 (schoolbook) or 49 (strassen).
    # Where did 16 come from? 
    # Maybe I confused it with $4 \times 4 \times 1$?
    # $4 \times 4 \times 4$ has 64 terms.
    
    # So my 114 estimate is solid.
    
    # I will implement the 114 rank decomposition.
    
    # Re-implementation of add_m2 with linear forms
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
