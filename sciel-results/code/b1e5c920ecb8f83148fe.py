"""Matrix multiplication 4,5,6 with Strassen-based rank reduction."""
import numpy as np

DIMENSIONS = [4, 5, 6]


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    
    Strategy:
    - Split the middle dimension 5 into 4+1
    - Split the output dimension 6 into 4+2
    - C = A @ B where A is 4x5, B is 5x6, C is 4x6
    - C[:, 0:4] = A[:, 0:4] @ B[0:4, 0:4] + A[:, 4:5] @ B[4:5, 0:4]
    - C[:, 4:6] = A[:, 0:4] @ B[0:4, 4:6] + A[:, 4:5] @ B[4:5, 4:6]
    
    Parts:
    1. A_top(4x4) @ B_tl(4x4): Strassen rank 49
    2. A_bot(4x1) @ B_bl(1x4): schoolbook rank 16
    3. A_top(4x4) @ B_tr(4x2): schoolbook rank 32
    4. A_bot(4x1) @ B_br(1x2): schoolbook rank 8
    
    Total rank = 49 + 16 + 32 + 8 = 105
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_sparse, v_sparse, w_sparse):
        """
        Add a term where u_sparse, v_sparse, w_sparse are dicts mapping index -> coefficient.
        u index: i*m+j for A[i,j]
        v index: j*p+k for B[j,k]  
        w index: k*n+i for C[i,k] (output convention W[k*n+i, r])
        """
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, coeff in u_sparse.items():
            u_vec[idx] = coeff
        for idx, coeff in v_sparse.items():
            v_vec[idx] = coeff
        for idx, coeff in w_sparse.items():
            w_vec[idx] = coeff
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Part 1: A_top(4x4) @ B_tl(4x4) using Strassen (rank 49)
    # A_top[i,j] = A[i,j] for i,j in 0..3
    # B_tl[j,k] = B[j,k] for j in 0..3, k in 0..3
    # C_tl[i,k] = C[i,k] for i in 0..3, k in 0..3
    
    # Strassen for 4x4@4x4: partition into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Block element indices (row, col) within the 4x4 matrix:
    # A00: (0,0),(0,1),(1,0),(1,1)
    # A01: (0,2),(0,3),(1,2),(1,3)
    # A10: (2,0),(2,1),(3,0),(3,1)
    # A11: (2,2),(2,3),(3,2),(3,3)
    
    def block_elements(base_row, base_col):
        """Get the 4 (row, col) positions in a 2x2 block starting at (base_row, base_col)."""
        return [(base_row + di, base_col + dj) for di in range(2) for dj in range(2)]
    
    # 2x2 Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
    # A indexed as [a00, a01, a10, a11], B as [b00, b01, b10, b11], C as [c00, c01, c10, c11]
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_2x2 = [
        # M1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = a(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = d(g-e)
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),
        # M5 = (a+c)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (d-c)(e+g)
        ([0, 0, -1, 1], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M7 = (b-d)(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Outer Strassen for 4x4@4x4:
    # Same structure but operating on 2x2 blocks
    outer_strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([0, 0, -1, 1], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    # For each outer term s, X_s and Y_s are 2x2 matrices (linear combos of blocks)
    # Then Z_s = X_s * Y_s computed with inner Strassen
    
    for s_idx in range(7):
        x_block_coeff = outer_strassen[s_idx][0]  # coeffs for A00,A01,A10,A11
        y_block_coeff = outer_strassen[s_idx][1]  # coeffs for B00,B01,B10,B11
        z_block_coeff = outer_strassen[s_idx][2]  # coeffs for C00,C01,C10,C11
        
        # X_s is a 2x2 matrix. X_s[pi,pj] = sum_b x_block_coeff[b] * A_block_b[pi,pj]
        # Y_s is a 2x2 matrix. Y_s[pi,pj] = sum_b y_block_coeff[b] * B_block_b[pi,pj]
        
        # For inner Strassen, we need X_s elements as linear combos of A[i,j]
        # and Y_s elements as linear combos of B[j,k]
        
        # X_s[pi,pj] where pi,pj in {0,1}:
        # A_block_b[pi,pj] where b in {0,1,2,3}:
        # b=0 (A00): row_base=0, col_base=0 -> (pi, pj)
        # b=1 (A01): row_base=0, col_base=2 -> (pi, pj+2)
        # b=2 (A10): row_base=2, col_base=0 -> (pi+2, pj)
        # b=3 (A11): row_base=2, col_base=2 -> (pi+2, pj+2)
        
        def x_s_element(pi, pj):
            """Return dict of A[i,j] -> coeff for X_s[pi,pj]"""
            result = {}
            for b in range(4):
                coeff = x_block_coeff[b]
                if coeff == 0:
                    continue
                row_base = (b // 2) * 2
                col_base = (b % 2) * 2
                i = row_base + pi
                j = col_base + pj
                key = i * m + j
                result[key] = result.get(key, 0) + coeff
            return result
        
        def y_s_element(pi, pj):
            """Return dict of B[j,k] -> coeff for Y_s[pi,pj]"""
            result = {}
            for b in range(4):
                coeff = y_block_coeff[b]
                if coeff == 0:
                    continue
                row_base = (b // 2) * 2
                col_base = (b % 2) * 2
                j = row_base + pi
                k = col_base + pj
                key = j * p + k
                result[key] = result.get(key, 0) + coeff
            return result
        
        # Now apply inner Strassen to X_s * Y_s
        for t_idx in range(7):
            a_coeffs = strassen_2x2[t_idx][0]  # for X_s elements [x00,x01,x10,x11]
            b_coeffs = strassen_2x2[t_idx][1]  # for Y_s elements [y00,y01,y10,y11]
            c_coeffs = strassen_2x2[t_idx][2]  # for Z_s elements [z00,z01,z10,z11]
            
            # A-part: sum_{pi,pj} a_coeffs[pi*2+pj] * X_s[pi,pj]
            a_part = {}
            for pi in range(2):
                for pj in range(2):
                    coeff = a_coeffs[pi * 2 + pj]
                    if coeff == 0:
                        continue
                    x_elem = x_s_element(pi, pj)
                    for a_key, a_coeff in x_elem.items():
                        total_coeff = coeff * a_coeff
                        a_part[a_key] = a_part.get(a_key, 0) + total_coeff
            
            # B-part: sum_{pi,pj} b_coeffs[pi*2+pj] * Y_s[pi,pj]
            b_part = {}
            for pi in range(2):
                for pj in range(2):
                    coeff = b_coeffs[pi * 2 + pj]
                    if coeff == 0:
                        continue
                    y_elem = y_s_element(pi, pj)
                    for b_key, b_coeff in y_elem.items():
                        total_coeff = coeff * b_coeff
                        b_part[b_key] = b_part.get(b_key, 0) + total_coeff
            
            # C-part: Z_s[pi,pj] contributes to C blocks with c_coeffs[pi*2+pj] * z_block_coeff[block]
            # Z_s[0,0] -> C00, Z_s[0,1] -> C01, Z_s[1,0] -> C10, Z_s[1,1] -> C11
            w_part = {}
            for pi in range(2):
                for pj in range(2):
                    c_coeff = c_coeffs[pi * 2 + pj]
                    if c_coeff == 0:
                        continue
                    block_idx = pi * 2 + pj  # 0=C00, 1=C01, 2=C10, 3=C11
                    z_coeff = z_block_coeff[block_idx]
                    if z_coeff == 0:
                        continue
                    total_coeff = c_coeff * z_coeff
                    # C block block_idx at position (pi,pj) within the block
                    # C00: row_base=0, col_base=0
                    # C01: row_base=0, col_base=2
                    # C10: row_base=2, col_base=0
                    # C11: row_base=2, col_base=2
                    row_base = (block_idx // 2) * 2
                    col_base = (block_idx % 2) * 2
                    i = row_base + pi
                    k = col_base + pj
                    w_key = k * n + i
                    w_part[w_key] = w_part.get(w_key, 0) + total_coeff
            
            # Skip if any part is empty
            if not a_part or not b_part or not w_part:
                continue
            
            # Remove zero coefficients
            a_part = {k: v for k, v in a_part.items() if v != 0}
            b_part = {k: v for k, v in b_part.items() if v != 0}
            w_part = {k: v for k, v in w_part.items() if v != 0}
            
            if a_part and b_part and w_part:
                add_term(a_part, b_part, w_part)
    
    # Part 2: A_bot(4x1) @ B_bl(1x4): schoolbook rank 16
    # A[i,4] * B[4,k] for i in 0..3, k in 0..3 -> C[i,k]
    for i in range(4):
        for k in range(4):
            u = {i * m + 4: 1}
            v = {4 * p + k: 1}
            w = {k * n + i: 1}
            add_term(u, v, w)
    
    # Part 3: A_top(4x4) @ B_tr(4x2): schoolbook rank 32
    # A[i,j] * B[j,k] for i in 0..3, j in 0..3, k in 4..5 -> C[i,k]
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u = {i * m + j: 1}
                v = {j * p + k: 1}
                w = {k * n + i: 1}
                add_term(u, v, w)
    
    # Part 4: A_bot(4x1) @ B_br(1x2): schoolbook rank 8
    # A[i,4] * B[4,k] for i in 0..3, k in 4..5 -> C[i,k]
    for i in range(4):
        for k in range(4, 6):
            u = {i * m + 4: 1}
            v = {4 * p + k: 1}
            w = {k * n + i: 1}
            add_term(u, v, w)
    
    rank = len(U_cols)
    U = [U_cols[r] for r in range(rank)]
    V = [V_cols[r] for r in range(rank)]
    W = [W_cols[r] for r in range(rank)]
    
    return {"U": U, "V": V, "W": W}
