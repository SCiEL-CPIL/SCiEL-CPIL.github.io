"""Matrix multiplication 4,5,6 with Strassen-based rank reduction (rank 105)."""
import numpy as np

DIMENSIONS = [4, 5, 6]


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    
    Strategy:
    - Split the middle dimension 5 into 4+1
    - Split the output dimension 6 into 4+2
    - C = A @ B where A is 4x5, B is 5x6, C is 4x6
    - C = A_top @ B_top + A_bottom @ B_bottom
    - A_top is 4x4, B_top is 4x6, A_bottom is 4x1, B_bottom is 1x6
    - Split B_top into B_top_left (4x4) and B_top_right (4x2)
    - C = A_top @ B_top_left + A_top @ B_top_right + A_bottom @ B_bottom
    - For A_top @ B_top_left (4x4 @ 4x4): use Strassen with rank 49
    - For A_top @ B_top_right (4x4 @ 4x2): schoolbook rank 32
    - For A_bottom @ B_bottom (4x1 @ 1x6): schoolbook rank 24
    - Total rank: 49 + 32 + 24 = 105
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a term where u_dict, v_dict, w_dict map indices to coefficients."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, coeff in u_dict.items():
            if coeff != 0:
                u_vec[idx] = coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                v_vec[idx] = coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                w_vec[idx] = coeff
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[c00, c01], [c10, c11]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+c)h, M6 = (d-c)(e+g), M7 = (b-d)(g+h)
    # C00 = M1 - M5 + M7, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
    
    strassen_2x2 = [
        # (A_coeffs for [a,b,c,d], B_coeffs for [e,f,g,h], C_coeffs for [C00,C01,C10,C11])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([0, 0, -1, 1], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    # Part 1: C_top_left = A_top(4x4) @ B_top_left(4x4), using Strassen (rank 49)
    # A_top[i,j] = A[i,j] for i,j in 0..3
    # B_top_left[j,k] = B[j,k] for j in 0..3, k in 0..3
    # C_top_left[i,k] = C[i,k] for i,k in 0..3
    
    # Block indices for 2x2 blocks within 4x4:
    # Block 0 (A00): rows 0-1, cols 0-1
    # Block 1 (A01): rows 0-1, cols 2-3
    # Block 2 (A10): rows 2-3, cols 0-1
    # Block 3 (A11): rows 2-3, cols 2-3
    
    def block_elements(block_idx):
        """Return list of (row, col) for elements in the given 2x2 block."""
        row_base = (block_idx // 2) * 2
        col_base = (block_idx % 2) * 2
        return [(row_base + di, col_base + dj) for di in range(2) for dj in range(2)]
    
    # Outer Strassen for 4x4 @ 4x4
    outer_strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([0, 0, -1, 1], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    # For each outer Strassen term, we have a 2x2 matrix product computed with inner Strassen
    for s_idx in range(7):
        x_coeff = outer_strassen[s_idx][0]  # Coefficients for A blocks [A00, A01, A10, A11]
        y_coeff = outer_strassen[s_idx][1]  # Coefficients for B blocks [B00, B01, B10, B11]
        z_coeff = outer_strassen[s_idx][2]  # Coefficients for C blocks [C00, C01, C10, C11]
        
        # X_s = sum_{b=0}^{3} x_coeff[b] * A_block[b]
        # Y_s = sum_{b=0}^{3} y_coeff[b] * B_block[b]
        
        # X_s is a 2x2 matrix where X_s[pi, pj] = sum_{b} x_coeff[b] * A_block[b][pi, pj]
        # Y_s is a 2x2 matrix where Y_s[pi, pj] = sum_{b} y_coeff[b] * B_block[b][pi, pj]
        
        # Inner Strassen: for each inner term t, we have:
        # A_part = sum_{pi,pj} a_t[pi,pj] * X_s[pi, pj]
        # B_part = sum_{pi,pj} b_t[pi,pj] * Y_s[pi, pj]
        # Product contributes to Z_s elements with c_t coefficients
        # Z_s elements contribute to C blocks with z_coeff
        
        for t_idx in range(7):
            a_t = strassen_2x2[t_idx][0]  # Coefficients for X_s elements [x00, x01, x10, x11]
            b_t = strassen_2x2[t_idx][1]  # Coefficients for Y_s elements [y00, y01, y10, y11]
            c_t = strassen_2x2[t_idx][2]  # Coefficients for Z_s elements [z00, z01, z10, z11]
            
            # A_part = sum_{pi,pj} a_t[pi*2+pj] * X_s[pi, pj]
            # = sum_{pi,pj} a_t[pi*2+pj] * sum_{b} x_coeff[b] * A_block[b][pi, pj]
            # = sum_{b, pi, pj} x_coeff[b] * a_t[pi*2+pj] * A_block[b][pi, pj]
            
            # Similarly for B_part
            
            # The product A_part * B_part contributes to C[i,k] with coefficient:
            # sum_{pi,pj} c_t[pi*2+pj] * z_coeff[block_of(i,k)] where (pi,pj) is the position in the 2x2 block
            
            # Let's collect all (A_index, B_index, C_index, coefficient) tuples
            a_dict = {}
            b_dict = {}
            w_dict = {}
            
            # Build A_part coefficients
            for pi in range(2):
                for pj in range(2):
                    coeff_t = a_t[pi * 2 + pj]
                    if coeff_t == 0:
                        continue
                    for b in range(4):
                        coeff_b = x_coeff[b]
                        if coeff_b == 0:
                            continue
                        total_coeff = coeff_t * coeff_b
                        if total_coeff == 0:
                            continue
                        row, col = block_elements(b)[pi * 2 + pj]
                        a_idx = row * m + col  # A[row, col] index in A (4x5)
                        a_dict[a_idx] = a_dict.get(a_idx, 0) + total_coeff
            
            # Build B_part coefficients
            for pi in range(2):
                for pj in range(2):
                    coeff_t = b_t[pi * 2 + pj]
                    if coeff_t == 0:
                        continue
                    for b in range(4):
                        coeff_b = y_coeff[b]
                        if coeff_b == 0:
                            continue
                        total_coeff = coeff_t * coeff_b
                        if total_coeff == 0:
                            continue
                        row, col = block_elements(b)[pi * 2 + pj]
                        b_idx = row * p + col  # B[row, col] index in B (5x6)
                        b_dict[b_idx] = b_dict.get(b_idx, 0) + total_coeff
            
            # Build W coefficients
            # The product contributes to Z_s elements, which contribute to C blocks
            for pi in range(2):
                for pj in range(2):
                    coeff_z = c_t[pi * 2 + pj]
                    if coeff_z == 0:
                        continue
                    # Z_s[pi, pj] contributes to C blocks
                    for cb in range(4):
                        coeff_cb = z_coeff[cb]
                        if coeff_cb == 0:
                            continue
                        total_coeff = coeff_z * coeff_cb
                        if total_coeff == 0:
                            continue
                        # C_block[cb] is a 2x2 block, and Z_s[pi,pj] is the (pi,pj) element
                        row, col = block_elements(cb)[pi * 2 + pj]
                        # C[row, col] index: W[col * n + row]
                        w_idx = col * n + row
                        w_dict[w_idx] = w_dict.get(w_idx, 0) + total_coeff
            
            # Check if this term is non-trivial
            if a_dict and b_dict and w_dict:
                add_term(a_dict, b_dict, w_dict)
    
    # Part 2: C_top_right = A_top(4x4) @ B_top_right(4x2), schoolbook rank 32
    # A_top[i,j] = A[i,j] for i in 0..3, j in 0..3
    # B_top_right[j,k] = B[j,k] for j in 0..3, k in 4..5
    # C_top_right[i,k] = C[i,k] for i in 0..3, k in 4..5
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                a_idx = i * m + j
                b_idx = j * p + k
                w_idx = k * n + i
                add_term({a_idx: 1}, {b_idx: 1}, {w_idx: 1})
    
    # Part 3: C_bottom = A_bottom(4x1) @ B_bottom(1x6), schoolbook rank 24
    # A_bottom[i,0] = A[i,4] for i in 0..3
    # B_bottom[0,k] = B[4,k] for k in 0..5
    # C_bottom[i,k] = C[i,k] for i in 0..3, k in 0..5
    
    for i in range(4):
        for k in range(6):
            a_idx = i * m + 4
            b_idx = 4 * p + k
            w_idx = k * n + i
            add_term({a_idx: 1}, {b_idx: 1}, {w_idx: 1})
    
    rank = len(U_cols)
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    return {"U": U, "V": V, "W": W}
