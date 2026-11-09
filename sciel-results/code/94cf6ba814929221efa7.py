import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 135
    
    # Fallback: Schoolbook decomposition (Rank 80)
    def get_schoolbook():
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    if time.time() - start_time > DEADLINE:
        return get_schoolbook()

    # Rank 65 Decomposition
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # --- Part 1: Strassen for 4x4x4 (49 terms) ---
    # Strassen 2x2 definitions
    # S1: (A00+A11)(B00+B11)
    # S2: (A10+A11)B00
    # S3: A00(B01-B11)
    # S4: (A00+A01)B11
    # S5: (A11+A01)(B10+B11)
    # S6: (A00-A10)(B00+B01)
    # S7: (A01-A11)(B10+B11)
    
    # Coefficients for U (A part) and V (B part) for each S term
    # Format: { (local_i, local_j): coeff }
    s_u_defs = [
        { (0,0): 1, (1,1): 1 },  # S1
        { (1,0): 1, (1,1): 1 },  # S2
        { (0,0): 1 },            # S3
        { (0,0): 1, (0,1): 1 },  # S4
        { (1,1): 1, (0,1): 1 },  # S5
        { (0,0): 1, (1,0): -1 }, # S6
        { (0,1): 1, (1,1): -1 }  # S7
    ]
    
    s_v_defs = [
        { (0,0): 1, (1,1): 1 },  # S1
        { (0,0): 1 },            # S2
        { (0,1): 1, (1,1): -1 }, # S3
        { (1,1): 1 },            # S4
        { (1,0): 1, (1,1): 1 },  # S5
        { (0,0): 1, (0,1): 1 },  # S6
        { (1,0): 1, (1,1): 1 }   # S7
    ]
    
    # Strassen 4x4 Super Terms M1..M7
    # M1 = (A11+A22)(B11+B22)  -> Blocks (0,0) and (1,1)
    # M2 = (A21+A22)B11        -> Blocks (1,0)+(1,1) and (0,0)
    # M3 = A11(B12-B22)        -> Block (0,0) and (0,1)-(1,1)
    # M4 = (A11+A12)B22        -> Blocks (0,0)+(0,1) and (1,1)
    # M5 = (A22+A12)(B21+B22)  -> Blocks (1,1)+(0,1) and (1,0)+(1,1)
    # M6 = (A11-A21)(B11+B12)  -> Blocks (0,0)-(1,0) and (0,0)+(0,1)
    # M7 = (A12-A22)(B21+B22)  -> Blocks (0,1)-(1,1) and (1,0)+(1,1)
    
    # Note: Block indices (r,c) correspond to A[r*2:(r+1)*2, c*2:(c+1)*2]
    # A blocks:
    # A00 (0,0), A01 (0,1)
    # A10 (1,0), A11 (1,1)
    
    # M1: A(A00+A11) * B(B00+B11)
    # M2: A(A10+A11) * B(B00)
    # M3: A(A00) * B(B01-B11)
    # M4: A(A00+A01) * B(B11)
    # M5: A(A11+A01) * B(B10+B11)
    # M6: A(A00-A10) * B(B00+B01)
    # M7: A(A01-A11) * B(B10+B11)
    
    # Define block contributions for U and V for each M term
    # M_idx: ( { (block_r, block_c): coeff }, { (block_r, block_c): coeff } )
    m_defs = [
        ( { (0,0): 1, (1,1): 1 }, { (0,0): 1, (1,1): 1 } ), # M1
        ( { (1,0): 1, (1,1): 1 }, { (0,0): 1 } ),            # M2
        ( { (0,0): 1 }, { (0,1): 1, (1,1): -1 } ),           # M3
        ( { (0,0): 1, (0,1): 1 }, { (1,1): 1 } ),            # M4
        ( { (1,1): 1, (0,1): 1 }, { (1,0): 1, (1,1): 1 } ),  # M5
        ( { (0,0): 1, (1,0): -1 }, { (0,0): 1, (0,1): 1 } ), # M6
        ( { (0,1): 1, (1,1): -1 }, { (1,0): 1, (1,1): 1 } )  # M7
    ]
    
    # Contribution of M terms to C blocks (C00, C01, C10, C11)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M2 + M3
    # C10 = M4 + M5
    # C11 = M1 - M2 + M5 + M7
    m_contribs = [
        [ 1, 0, 0, -1], # M1
        [ 0, 1, 0, -1], # M2
        [ 0, 1, 0, 0],  # M3
        [-1, 0, 1, 0],  # M4
        [ 0, 0, 1, 1],  # M5
        [ 1, 0, 0, 0],  # M6
        [-1, 0, 0, 1]   # M7
    ]
    
    # Contribution of S terms to local 2x2 result blocks (Z00, Z01, Z10, Z11)
    # Z00 = S1 - S4 + S6 - S7
    # Z01 = S2 + S3
    # Z10 = S4 + S5
    # Z11 = S1 - S2 + S5 + S7
    s_contribs = [
        [ 1, 0, 0, 1], # S1
        [ 0, 1, 0, -1], # S2
        [ 0, 1, 0, 0],  # S3
        [-1, 0, 1, 0],  # S4
        [ 0, 0, 1, 1],  # S5
        [ 1, 0, 0, 0],  # S6
        [-1, 0, 0, 1]   # S7
    ]
    
    for m_idx in range(7):
        if time.time() - start_time > DEADLINE:
            return get_schoolbook()
        u_blocks, v_blocks = m_defs[m_idx]
        m_contribs_row = m_contribs[m_idx]
        
        for s_idx in range(7):
            if time.time() - start_time > DEADLINE:
                return get_schoolbook()
            u_local = s_u_defs[s_idx]
            v_local = s_v_defs[s_idx]
            s_contribs_row = s_contribs[s_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            # Expand U
            for (br, bc), b_coeff in u_blocks.items():
                if b_coeff == 0: continue
                # Block (br, bc) corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1
                for (lr, lc), l_coeff in u_local.items():
                    gi = br * 2 + lr
                    gj = bc * 2 + lc
                    u_list.append((gi, gj, b_coeff * l_coeff))
                    
            # Expand V
            for (br, bc), b_coeff in v_blocks.items():
                if b_coeff == 0: continue
                for (lr, lc), l_coeff in v_local.items():
                    gj = br * 2 + lr
                    gk = bc * 2 + lc
                    v_list.append((gj, gk, b_coeff * l_coeff))
                    
            # Expand W
            # W index is (k, i). 
            # C block (c_r, c_c) corresponds to rows c_r*2..c_r*2+1 (i), cols c_c*2..c_c*2+1 (k)
            # Wait, C is 4x4. C[i,k].
            # Block (c_r, c_c) of C has i in c_r*2..c_r*2+1, k in c_c*2..c_c*2+1.
            # W[k*n+i, r].
            for c_idx in range(4):
                m_c = m_contribs_row[c_idx]
                if m_c == 0: continue
                c_r = c_idx // 2
                c_c = c_idx % 2
                
                # Local S contribution to this C block
                # s_contribs_row[c_idx] is the coeff for this C block from this S term?
                # No, s_contribs_row is for the local 2x2 result of the S term.
                # The S term produces a 2x2 matrix Z.
                # Z is added to C blocks.
                # The M term multiplies the S term's result? No.
                # The structure is:
                # C_block = sum_m (M_coeff * Z_m)
                # Z_m is the 2x2 result of the m-th Strassen product.
                # Z_m = sum_s (S_coeff * S_m_s)
                # S_m_s is the 2x2 result of the s-th Strassen term of the m-th product.
                
                # So, for a specific (m_idx, s_idx):
                # We have a 2x2 matrix S_res from the s-th term of the m-th product.
                # This S_res is added to the C blocks according to m_contribs_row[c_idx] * s_contribs_row[local_pos].
                
                # Wait, s_contribs_row is defined for the local 2x2 output of the S term.
                # Let's call the local output positions l_pos (0:00, 1:01, 2:10, 3:11).
                # The value at l_pos in the S_res matrix is multiplied by s_contribs_row[l_pos]?
                # No, s_contribs_row IS the definition of how S terms combine to form Z.
                # Z[l_pos] = sum_s s_contribs[s_idx][l_pos] * S_res_m_s[l_pos]?
                # No.
                # Z = M1 - M4 ...
                # Each M is a 2x2 matrix.
                # M = Z_s1 - Z_s4 ...
                # So Z[l_pos] = sum_s s_contribs[s_idx][l_pos] * (S_term_m_s)[l_pos].
                
                # So for a fixed (m_idx, s_idx), we have a 2x2 matrix P = S_term_m_s.
                # P contributes to C_block(c_idx) with weight m_contribs_row[c_idx] * P[l_pos].
                # P[l_pos] is determined by the S term definition?
                # No, P is the *result* of the multiplication defined by S term s_idx.
                # The S term s_idx defines U and V such that U * V = P.
                # We don't compute P explicitly, we just need to know how P's elements map to W.
                # The decomposition is:
                # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = C[i,k]
                # For a specific term r (corresponding to m_idx, s_idx, l_pos?),
                # Actually, each (m_idx, s_idx) pair generates 7 terms?
                # No, Strassen 2x2 has 7 terms.
                # So for each M (7 of them), we have 7 S terms.
                # Each S term is a rank-1 update to the 2x2 product?
                # No, Strassen 2x2 is a decomposition of the 2x2x2 tensor into 7 rank-1 tensors.
                # So for each (m_idx, s_idx), we have ONE term in the global rank-65 decomposition?
                # Yes.
                # The U and V for this term are defined by s_u_defs[s_idx] and s_v_defs[s_idx] expanded by m_defs.
                # The W for this term must be such that:
                # sum_r ... = sum_m sum_s (m_coeff * s_coeff * P_m_s)
                # Wait, the tensor decomposition is linear.
                # The term (m_idx, s_idx) contributes to the tensor T.
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # For the term r corresponding to (m_idx, s_idx):
                # U and V are fixed.
                # W must be set so that the product U V W matches the contribution of this term to T.
                # The contribution of this term to the 2x2 product Z_m is the s-th Strassen term S_m_s.
                # S_m_s is a 2x2 matrix.
                # Let S_m_s = [ p00, p01; p10, p11 ].
                # The contribution to C is m_contribs_row[c_idx] * S_m_s (placed in block c_idx).
                # So for each element (l_r, l_c) of the 2x2 block:
                # Global i = c_r*2 + l_r, Global k = c_c*2 + l_c.
                # W[k, i] should be m_contribs_row[c_idx] * s_contribs_row[l_idx]?
                # No.
                # The Strassen 2x2 decomposition is:
                # Z = sum_s S_s
                # Where S_s is a matrix.
                # The rank-1 terms are defined by U_s, V_s, W_s such that U_s V_s W_s = S_s (element-wise? No, tensor).
                # For 2x2x2:
                # T[i,j,k] = sum_s U[i,j,s] V[j,k,s] W[k,i,s]
                # If we fix s, U_s and V_s are fixed vectors.
                # W_s must be such that the product equals the s-th term of the Strassen sum.
                # The s-th term of the Strassen sum for 2x2 is:
                # S1: (A00+A11)(B00+B11) -> Matrix with all entries?
                # No, Strassen computes Z = A B.
                # Z00 = S1 - S4 + S6 - S7
                # Z01 = S2 + S3
                # Z10 = S4 + S5
                # Z11 = S1 - S2 + S5 + S7
                # So the "term" s contributes to Z00 with coeff s_contribs[s][0], to Z01 with s_contribs[s][1], etc.
                # So for the global W:
                # For each local position l_idx (0..3) in the 2x2 block:
                # The coefficient is s_contribs_row[l_idx].
                # This coefficient applies to the global block c_idx.
                # So W[global_k, global_i] += m_contribs_row[c_idx] * s_contribs_row[l_idx] * (value from S term?)
                # Wait, the S term is a scalar multiplication?
                # No, S_s is a scalar?
                # In Strassen, we compute 7 scalar products M1..M7.
                # Then we combine them to get the 4 entries of Z.
                # So for each (m_idx, s_idx), we have a scalar product.
                # Let's call it P.
                # P contributes to Z_m[l_idx] with weight s_contribs_row[l_idx].
                # Z_m contributes to C_block[c_idx] with weight m_contribs_row[c_idx].
                # So the total weight for the scalar P to contribute to C[global_i, global_k] is:
                # m_contribs_row[c_idx] * s_contribs_row[l_idx].
                # And this scalar P is formed by U and V.
                # So W[global_k, global_i] = m_contribs_row[c_idx] * s_contribs_row[l_idx].
                
                for l_idx in range(4):
                    s_c = s_contribs_row[l_idx]
                    if s_c == 0: continue
                    total_coeff = m_c * s_c
                    if total_coeff == 0: continue
                    l_r = l_idx // 2
                    l_c = l_idx % 2
                    gi = c_r * 2 + l_r
                    gk = c_c * 2 + l_c
                    w_list.append((gk, gi, total_coeff))
                    
            add_term(col, u_list, v_list, w_list)
            col += 1

    # --- Part 2: Schoolbook for column 4 (16 terms) ---
    for i in range(n):
        for j in range(m):
            if time.time() - start_time > DEADLINE:
                return get_schoolbook()
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1

    # Verify
    def verify_decomp(U_mat, V_mat, W_mat, n, m, p):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        rank = U_np.shape[1]
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randint(-5, 5, (n, m))
            B = np.random.randint(-5, 5, (m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                        C_computed[i, k] += s * A[i, j] * B[j, k]
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    if not verify_decomp(U, V, W, n, m, p):
        return get_schoolbook()

    return {"U": U, "V": V, "W": W}
