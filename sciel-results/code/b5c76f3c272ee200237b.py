import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=5
    
    # We will build U, V, W matrices.
    # Rank is determined by the number of terms.
    # We'll store the terms as (u_vec, v_vec, w_vec) or directly fill the matrices.
    
    # Since we don't know the final rank until we count, we'll use lists of vectors
    # or just build the matrices dynamically.
    # Max possible rank is 75.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        # u_idx is index in U (0 to n*m-1)
        # v_idx is index in V (0 to m*p-1)
        # w_idx is index in W (0 to p*n-1)
        # We need to align these into the same column.
        # Since we are building column by column, we just append.
        pass

    # Actually, it's easier to build the full matrices.
    # We'll determine the total rank first, then fill.
    # Or just append columns.
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    # Partitions
    # I: 0,1 (size 2), 2 (size 1)
    # J: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    # K: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    
    I_blocks = [(0, 2), (2, 3)]
    J_blocks = [(0, 2), (2, 4), (4, 5)]
    K_blocks = [(0, 2), (2, 4), (4, 5)]
    
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    
                    u[i*m + j] = 1
                    v[j*p + k] = 1
                    w[k*n + i] = 1
                    
                    cols_U.append(u)
                    cols_V.append(v)
                    cols_W.append(w)
                    
    def add_strassen_block(i_start, i_end, j_start, j_end, k_start, k_end):
        # i_end - i_start == 2, j_end - j_start == 2, k_end - k_start == 2
        # Map local indices 0,1 to global
        # A[0,0] = A[i_start, j_start]
        # A[0,1] = A[i_start, j_start+1]
        # A[1,0] = A[i_start+1, j_start]
        # A[1,1] = A[i_start+1, j_start+1]
        
        # Strassen Terms:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # We need to define the linear combinations for U, V, W.
        # U corresponds to A, V to B, W to C.
        # The term is sum of products of linear combos of U, V, W.
        # For each Strassen multiplication M_t, we have:
        # M_t = U_t^T A V_t  (conceptually, but U,V,W are vectors)
        # Actually, the tensor decomposition is:
        # T = sum_t U_t \otimes V_t \otimes W_t
        # For Strassen, we express the 8 products as 7 terms.
        
        # Let's define the 7 terms explicitly.
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        def make_vec(indices, coeffs):
            vec = [0] * (n*m)
            for idx, coeff in zip(indices, coeffs):
                vec[idx] += coeff
            return vec
        
        def make_vec_v(indices, coeffs):
            vec = [0] * (m*p)
            for idx, coeff in zip(indices, coeffs):
                vec[idx] += coeff
            return vec
            
        def make_vec_w(indices, coeffs):
            vec = [0] * (p*n)
            for idx, coeff in zip(indices, coeffs):
                vec[idx] += coeff
            return vec

        # U indices
        u_00 = i_start*m + j_start
        u_01 = i_start*m + (j_start+1)
        u_10 = (i_start+1)*m + j_start
        u_11 = (i_start+1)*m + (j_start+1)
        
        # V indices
        v_00 = j_start*p + k_start
        v_01 = j_start*p + (k_start+1)
        v_10 = (j_start+1)*p + k_start
        v_11 = (j_start+1)*p + (k_start+1)
        
        # W indices
        w_00 = k_start*n + i_start
        w_01 = k_start*n + (i_start+1)
        w_10 = (k_start+1)*n + i_start
        w_11 = (k_start+1)*n + (i_start+1)
        
        # Term 1: (A00 + A11)(B00 + B11)
        # C00 += M1, C01 += M1, C10 += M1, C11 -= M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 columns.
        
        # 1. M1
        u1 = make_vec([u_00, u_11], [1, 1])
        v1 = make_vec_v([v_00, v_11], [1, 1])
        # W contributions:
        # C00: +1, C01: 0, C10: 0, C11: -1? No, check Strassen formulas.
        # Standard Strassen:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B21-B22)
        # P4 = A22(B12-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # My indices:
        # i0, i1 correspond to 1, 2
        # j0, j1 correspond to 1, 2
        # k0, k1 correspond to 1, 2
        
        # C00 (i0,k0) gets P1
        # C01 (i0,k1) gets nothing from P1?
        # Wait, P1 contributes to C11, C22?
        # C11 = P1 + ...
        # C22 = P1 + ...
        # So P1 contributes to C00 and C11 (in 0-indexed 2x2).
        
        w1 = make_vec_w([w_00, w_11], [1, 1])
        cols_U.append(u1); cols_V.append(v1); cols_W.append(w1)
        
        # 2. M2 = (A10 + A11)B00
        u2 = make_vec([u_10, u_11], [1, 1])
        v2 = make_vec_v([v_00], [1])
        # C10 (i1,k0) gets P2. C11 (i1,k1) gets -P2.
        w2 = make_vec_w([w_10, w_11], [1, -1])
        cols_U.append(u2); cols_V.append(v2); cols_W.append(w2)
        
        # 3. M3 = A00(B10 - B11)
        u3 = make_vec([u_00], [1])
        v3 = make_vec_v([v_10, v_11], [1, -1])
        # C01 (i0,k1) gets P3. C11 (i1,k1) gets P3.
        w3 = make_vec_w([w_01, w_11], [1, 1])
        cols_U.append(u3); cols_V.append(v3); cols_W.append(w3)
        
        # 4. M4 = A11(B01 - B00)
        u4 = make_vec([u_11], [1])
        v4 = make_vec_v([v_01, v_00], [1, -1])
        # C00 (i0,k0) gets P4. C10 (i1,k0) gets P4.
        w4 = make_vec_w([w_00, w_10], [1, 1])
        cols_U.append(u4); cols_V.append(v4); cols_W.append(w4)
        
        # 5. M5 = (A00 + A01)B11
        u5 = make_vec([u_00, u_01], [1, 1])
        v5 = make_vec_v([v_11], [1])
        # C00 (i0,k0) gets -P5. C01 (i0,k1) gets P5.
        w5 = make_vec_w([w_00, w_01], [-1, 1])
        cols_U.append(u5); cols_V.append(v5); cols_W.append(w5)
        
        # 6. M6 = (A10 - A00)(B00 + B01)
        u6 = make_vec([u_10, u_00], [1, -1])
        v6 = make_vec_v([v_00, v_01], [1, 1])
        # C11 (i1,k1) gets P6.
        w6 = make_vec_w([w_11], [1])
        cols_U.append(u6); cols_V.append(v6); cols_W.append(w6)
        
        # 7. M7 = (A01 - A11)(B10 + B11)
        u7 = make_vec([u_01, u_11], [1, -1])
        v7 = make_vec_v([v_10, v_11], [1, 1])
        # C00 (i0,k0) gets P7.
        w7 = make_vec_w([w_00], [1])
        cols_U.append(u7); cols_V.append(v7); cols_W.append(w7)

    # Iterate through all blocks
    for i_s, i_e in I_blocks:
        for j_s, j_e in J_blocks:
            for k_s, k_e in K_blocks:
                if (i_e - i_s == 2) and (j_e - j_s == 2) and (k_e - k_s == 2):
                    add_strassen_block(i_s, i_e, j_s, j_e, k_s, k_e)
                else:
                    add_schoolbook_block(i_s, i_e, j_s, j_e, k_s, k_e)
                    
    # Convert lists of lists to matrices
    # The evaluator expects {'U': U, 'V': V, 'W': W}
    # U shape (n*m, rank)
    # V shape (m*p, rank)
    # W shape (p*n, rank)
    
    rank = len(cols_U)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c in range(rank):
        for i in range(n*m):
            if cols_U[c][i] != 0:
                U[i, c] = cols_U[c][i]
        for i in range(m*p):
            if cols_V[c][i] != 0:
                V[i, c] = cols_V[c][i]
        for i in range(p*n):
            if cols_W[c][i] != 0:
                W[i, c] = cols_W[c][i]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
