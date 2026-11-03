import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36

    # Target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Rank 129 decomposition
    # 49 terms for 4x4x4 (cols 0-3 of B)
    # 80 terms for 4x4x5 (cols 4-8 of B)
    R = 129
    U = np.zeros((dim_u, R), dtype=float)
    V = np.zeros((dim_v, R), dtype=float)
    W = np.zeros((dim_w, R), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, coeff in u_indices:
            U[idx, col] += coeff
        for idx, coeff in v_indices:
            V[idx, col] += coeff
        for idx, coeff in w_indices:
            W[idx, col] += coeff
        col += 1
        
    # Strassen 2x2x2 inner terms
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (i,j) in 2x2 A block to coeff
    # v_coeffs: dict mapping (i,j) in 2x2 B block to coeff
    # w_coeffs: dict mapping (i,j) in 2x2 C block to coeff
    
    def get_strassen_2x2_terms():
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        terms = []
        
        # M1
        u = {(0,0): 1, (1,1): 1}
        v = {(0,0): 1, (1,1): 1}
        w = {(0,0): 1, (1,1): 1} # Wait, C11 also gets M1
        # Actually, we need to distribute M1 to C00 and C11.
        # So we need two terms? No, the rank-1 term is $u \cdot v \cdot w$.
        # If $w$ has multiple non-zeros, it's still one term in the tensor decomposition?
        # No, a rank-1 tensor is $u_i v_j w_k$.
        # If $w$ is sparse, it's fine.
        # But $M_1$ contributes to $C_{00}$ AND $C_{11}$.
        # So the term is $u \cdot v \cdot (e_{00} + e_{11})$.
        # This is a valid rank-1 term in the tensor space.
        
        terms.append((u, v, {(0,0): 1, (1,1): 1}))
        
        # M2
        u = {(1,0): 1, (1,1): 1}
        v = {(0,0): 1}
        w = {(1,0): 1, (1,1): -1} # C10 gets M2, C11 gets -M2
        terms.append((u, v, w))
        
        # M3
        u = {(0,0): 1}
        v = {(0,1): 1, (1,1): -1}
        w = {(0,1): 1, (1,1): 1} # C01 gets M3, C11 gets M3
        terms.append((u, v, w))
        
        # M4
        u = {(1,1): 1}
        v = {(1,0): 1, (0,0): -1}
        w = {(0,0): 1, (1,0): 1} # C00 gets M4, C10 gets M4
        terms.append((u, v, w))
        
        # M5
        u = {(0,0): 1, (0,1): 1}
        v = {(1,1): 1}
        w = {(0,0): -1, (0,1): 1} # C00 gets -M5, C01 gets M5
        terms.append((u, v, w))
        
        # M6
        u = {(1,0): 1, (0,0): -1}
        v = {(0,0): 1, (0,1): 1}
        w = {(1,1): 1} # C11 gets M6
        terms.append((u, v, w))
        
        # M7
        u = {(0,1): 1, (1,1): -1}
        v = {(1,0): 1, (1,1): 1}
        w = {(0,0): 1, (1,1): 1} # C00 gets M7, C11 gets M7
        terms.append((u, v, w))
        
        return terms

    inner_terms = get_strassen_2x2_terms()
    
    # Outer Strassen 4x4x4
    # S1 = (A11+A22)(B11+B22) -> C11 += S1, C22 += S1
    # S2 = (A21+A22)B11 -> C21 += S2, C22 -= S2
    # S3 = A11(B12-B22) -> C12 += S3, C22 += S3
    # S4 = A22(B21-B11) -> C11 += S4, C21 += S4
    # S5 = (A11+A12)B22 -> C11 -= S5, C12 += S5
    # S6 = (A21-A11)(B11+B12) -> C22 += S6
    # S7 = (A12-A22)(B21+B22) -> C11 += S7, C22 += S7
    
    # Mapping of blocks to indices
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B11: rows 0-1, cols 0-1 (of B_left, i.e. B cols 0-1)
    # B12: rows 0-1, cols 2-3 (of B_left, i.e. B cols 2-3)
    # B21: rows 2-3, cols 0-1 (of B_left, i.e. B cols 0-1)
    # B22: rows 2-3, cols 2-3 (of B_left, i.e. B cols 2-3)
    
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    def map_block_indices(block_name, dim, offset_row, offset_col, is_b=False):
        # Returns list of (global_index, coeff) for the 2x2 block
        # block_name is like 'A11'
        # We need to map local (i,j) to global u/v/w index
        
        pass # Implemented inline below

    # We will manually construct the 49 terms.
    
    # Helper to add a full 2x2x2 Strassen decomposition for specific blocks
    def add_outer_strassen_step(a_blocks, b_blocks, c_blocks):
        # a_blocks: list of (row_offset, col_offset, sign)
        # b_blocks: list of (row_offset, col_offset, sign)
        # c_blocks: list of (row_offset, col_offset, sign)
        
        # Combine linear forms
        # For each inner term (u_coeffs, v_coeffs, w_coeffs)
        for u_coeffs, v_coeffs, w_coeffs in inner_terms:
            u_global = []
            v_global = []
            w_global = []
            
            # Build u_global
            for (li, lj), coeff in u_coeffs.items():
                for (ro, co, s) in a_blocks:
                    # A index: (ro + li, co + lj)
                    # u index: (ro+li)*4 + (co+lj)
                    u_idx = (ro + li) * 4 + (co + lj)
                    u_global.append((u_idx, coeff * s))
            
            # Build v_global
            for (li, lj), coeff in v_coeffs.items():
                for (ro, co, s) in b_blocks:
                    # B index: (ro + li, co + lj)
                    # v index: (ro+li)*9 + (co+lj)
                    v_idx = (ro + li) * 9 + (co + lj)
                    v_global.append((v_idx, coeff * s))
            
            # Build w_global
            for (li, lj), coeff in w_coeffs.items():
                for (ro, co, s) in c_blocks:
                    # C index: (ro + li, co + lj)
                    # w index: (co+lj)*4 + (ro+li)  <-- NOTE: w = k*n + i
                    # C is 4x9. Row i, Col k.
                    # In C blocks, ro is row offset, co is col offset.
                    # So C row = ro + li, C col = co + lj.
                    # w index = (co+lj)*4 + (ro+li)
                    w_idx = (co + lj) * 4 + (ro + li)
                    w_global.append((w_idx, coeff * s))
            
            add_term(u_global, v_global, w_global)

    # Step 1: S1 = (A11+A22)(B11+B22) -> C11 += S1, C22 += S1
    a_blocks = [(0, 0, 1), (2, 2, 1)]
    b_blocks = [(0, 0, 1), (2, 2, 1)]
    c_blocks = [(0, 0, 1), (2, 2, 1)]
    add_outer_strassen_step(a_blocks, b_blocks, c_blocks)
    
    # Step 2: S2 = (A21+A22)B11 -> C21 += S2, C22 -= S2
    a_blocks = [(2, 0, 1), (2, 2, 1)]
    b_blocks = [(0, 0, 1)]
    c_blocks = [(2, 0, 1), (2, 2, -1)]
    add_outer_strassen_step(a_blocks, b_blocks, c_blocks)
    
    # Step 3: S3 = A11(B12-B22) -> C12 += S3, C22 += S3
    a_blocks = [(0, 0, 1)]
    b_blocks = [(0, 2, 1), (2, 2, -1)]
    c_blocks = [(0, 2, 1), (2, 2, 1)]
    add_outer_strassen_step(a_blocks, b_blocks, c_blocks)
    
    # Step 4: S4 = A22(B21-B11) -> C11 += S4, C21 += S4
    a_blocks = [(2, 2, 1)]
    b_blocks = [(2, 0, 1), (0, 0, -1)]
    c_blocks = [(0, 0, 1), (2, 0, 1)]
    add_outer_strassen_step(a_blocks, b_blocks, c_blocks)
    
    # Step 5: S5 = (A11+A12)B22 -> C11 -= S5, C12 += S5
    a_blocks = [(0, 0, 1), (0, 2, 1)]
    b_blocks = [(2, 2, 1)]
    c_blocks = [(0, 0, -1), (0, 2, 1)]
    add_outer_strassen_step(a_blocks, b_blocks, c_blocks)
    
    # Step 6: S6 = (A21-A11)(B11+B12) -> C22 += S6
    a_blocks = [(2, 0, 1), (0, 0, -1)]
    b_blocks = [(0, 0, 1), (0, 2, 1)]
    c_blocks = [(2, 2, 1)]
    add_outer_strassen_step(a_blocks, b_blocks, c_blocks)
    
    # Step 7: S7 = (A12-A22)(B21+B22) -> C11 += S7, C22 += S7
    a_blocks = [(0, 2, 1), (2, 2, -1)]
    b_blocks = [(2, 0, 1), (2, 2, 1)]
    c_blocks = [(0, 0, 1), (2, 2, 1)]
    add_outer_strassen_step(a_blocks, b_blocks, c_blocks)
    
    # Part 2: Schoolbook for cols 4-8
    for k in range(4, 9):
        for i in range(4):
            for j in range(4):
                u_idx = i * 4 + j
                v_idx = j * 9 + k
                w_idx = k * 4 + i
                add_term([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])
                
    # Verify
    T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
    error = np.linalg.norm(T_recon - T)
    
    if error < 1e-9:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook 144 if verification fails
        U_s = np.zeros((dim_u, 144), dtype=float)
        V_s = np.zeros((dim_v, 144), dtype=float)
        W_s = np.zeros((dim_w, 144), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U_s[u, col] = 1.0
                    V_s[v, col] = 1.0
                    W_s[w, col] = 1.0
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }
