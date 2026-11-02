import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        for row, val in u_dict.items():
            u_col[row] += val
        v_col = [0] * (m * p)
        for row, val in v_dict.items():
            v_col[row] += val
        w_col = [0] * (p * n)
        for row, val in w_dict.items():
            w_col[row] += val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_4x4_x_4x4(b_col_offset):
        """
        Strassen decomposition for A(4x4) * B_block(4x4) where B_block starts at column b_col_offset.
        Adds 28 terms.
        """
        # Strassen products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each Mi = Xi * Yi where Xi and Yi are 2x2 matrices:
        # Xi[i',j'] = sum of A-block entries with coefficients
        # Yi[j',k'] = sum of B-block entries with coefficients
        
        # Schoolbook decomposition of Xi*Yi:
        # For s in {0,1}, k in {0,1}:
        #   U: Xi[:,s]
        #   V: Yi[s,:]
        #   W: e_s * e_k^T (in local 2x2 coordinates)
        
        # Define the 7 products as (Xi_coeffs, Yi_coeffs, C_block)
        # Xi_coeffs: dict from (i',j') to list of (A_block, coeff)
        # Yi_coeffs: dict from (j',k') to list of (B_block, coeff)
        # C_block: (c_row_offset, c_col_offset) for the 2x2 C block
        
        # A blocks: A11=(0,0), A12=(0,2), A21=(2,0), A22=(2,2)
        # B blocks: B11=(0,0), B12=(0,2), B21=(2,0), B22=(2,2) (relative to b_col_offset)
        
        # M1: Xi = A11+A22, Yi = B11+B22, C = C11
        # M2: Xi = A21+A22, Yi = B11, C = C21
        # M3: Xi = A11, Yi = B12-B22, C = C12
        # M4: Xi = A22, Yi = B21-B11, C = C22
        # M5: Xi = A11+A12, Yi = B22, C = C12
        # M6: Xi = A21-A11, Yi = B11+B12, C = C21
        # M7: Xi = A12-A22, Yi = B21+B22, C = C22
        
        products = [
            # (Xi, Yi, C_block)
            # Xi: list of (a_row_offset, a_col_offset, coeff)
            # Yi: list of (b_row_offset, b_col_offset, coeff)
            # C_block: (c_row_offset, c_col_offset)
            (
                [(0,0,1), (2,2,1)],  # A11 + A22
                [(0,0,1), (2,2,1)],  # B11 + B22
                (0,0)  # C11
            ),
            (
                [(2,0,1), (2,2,1)],  # A21 + A22
                [(0,0,1)],  # B11
                (2,0)  # C21
            ),
            (
                [(0,0,1)],  # A11
                [(0,2,1), (2,2,-1)],  # B12 - B22
                (0,2)  # C12
            ),
            (
                [(2,2,1)],  # A22
                [(2,0,1), (0,0,-1)],  # B21 - B11
                (2,2)  # C22
            ),
            (
                [(0,0,1), (0,2,1)],  # A11 + A12
                [(2,2,1)],  # B22
                (0,2)  # C12
            ),
            (
                [(2,0,1), (0,0,-1)],  # A21 - A11
                [(0,0,1), (0,2,1)],  # B11 + B12
                (2,0)  # C21
            ),
            (
                [(0,2,1), (2,2,-1)],  # A12 - A22
                [(2,0,1), (2,2,1)],  # B21 + B22
                (2,2)  # C22
            ),
        ]
        
        for Xi, Yi, (c_row_off, c_col_off) in products:
            # Xi is a 2x2 matrix: Xi[i',j'] = sum of A[c_row_off+i', c_col_off+j'] * coeff
            # Yi is a 2x2 matrix: Yi[j',k'] = sum of B[b_row_off+j', b_col_off+k'] * coeff
            # C block is C[c_row_off+i', c_col_off+k']
            
            # Schoolbook decomposition of Xi*Yi:
            for s in range(2):
                for k in range(2):
                    u_dict = {}
                    v_dict = {}
                    w_dict = {}
                    
                    # U: Xi[:,s]
                    for (a_row, a_col, coeff) in Xi:
                        for i_prime in range(2):
                            # Xi[i',s] includes A[a_row+i', a_col+s] * coeff
                            # But only if a_col+s is a valid column in the 2x2 block
                            # Actually, Xi is defined as: Xi[i',j'] = sum of A[a_row+i', a_col+j'] * coeff
                            # So Xi[i',s] = sum of A[a_row+i', a_col+s] * coeff
                            u_row = (a_row + i_prime) * 4 + (a_col + s)
                            u_dict[u_row] = u_dict.get(u_row, 0) + coeff
                    
                    # V: Yi[s,:]
                    for (b_row, b_col, coeff) in Yi:
                        for k_prime in range(2):
                            # Yi[s,k'] = sum of B[b_row+s, b_col+k'] * coeff
                            v_row = (b_row + s) * 9 + (b_col + k_prime + b_col_offset)
                            v_dict[v_row] = v_dict.get(v_row, 0) + coeff
                    
                    # W: e_s * e_k^T in local coordinates
                    # C[c_row_off+i', c_col_off+k] for i' in {0,1}
                    for i_prime in range(2):
                        w_row = (c_col_off + k + b_col_offset) * 4 + (c_row_off + i_prime)
                        w_dict[w_row] = w_dict.get(w_row, 0) + 1
                    
                    add_term(u_dict, v_dict, w_dict)
    
    def schoolbook_4x4_x_4x1(b_col_offset):
        """4x4 x 4x1 schoolbook, rank 16."""
        for i in range(4):
            for j in range(4):
                add_term(
                    {i * 4 + j: 1},
                    {j * 9 + b_col_offset: 1},
                    {(b_col_offset) * 4 + i: 1}
                )
    
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    strassen_4x4_x_4x4(0)  # A * B1, B1 is columns 0-3
    strassen_4x4_x_4x4(4)  # A * B2, B2 is columns 4-7
    schoolbook_4x4_x_4x1(8)  # A * B3, B3 is column 8
    
    U = list(zip(*U_cols)) if U_cols else []
    V = list(zip(*V_cols)) if V_cols else []
    W = list(zip(*W_cols)) if W_cols else []
    
    return {"U": U, "V": V, "W": W}
