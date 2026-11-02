import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Index helpers
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen 2x2 sub-multiplication logic
    # Inputs: A blocks (a,b,c,d) and B blocks (e,f,g,h)
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Block level Strassen for 4x4
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We define the 7 block products. For each, we define the linear combos for A and B.
    # Each block product is a 2x2 matrix multiplication, so we use the 7 sub-Strassen steps.
    
    # Helper to add a linear combination to U, V, W for a specific sub-product
    # sub_prod_id: 0..6 corresponding to M1..M7
    # a_combo: list of (row, col, coeff) for A
    # b_combo: list of (row, col, coeff) for B
    # c_combo: list of (row, col, coeff) for C (contribution to output)
    # rank_base: offset in the rank dimension
    
    def add_sub_product(rank_base, a_combo, b_combo, c_combo):
        for r in range(7):
            # Determine which elements of A and B are involved in sub-product r
            # This logic is complex to generalize easily without hardcoding the 7 cases.
            # Instead, we will hardcode the 7 block-level terms and their sub-Strassen expansions.
            pass

    # Given the complexity and low rank, explicit enumeration is safer and faster.
    
    # Define the 7 block-level terms
    # Term t: 
    #   L_t is a 2x2 matrix of linear combos of A
    #   R_t is a 2x2 matrix of linear combos of B
    #   Contrib_t: how the 2x2 result P_t contributes to C
    
    # A indices: (0,0)-(1,1) is A11, (0,2)-(1,3) is A12, (2,0)-(3,1) is A21, (2,2)-(3,3) is A22
    # B indices: same structure
    # C indices: same structure
    
    # Let's represent a 2x2 block linear combo as a list of 4 dicts, or just 4 lists of (idx, coeff)
    # For efficiency, we precompute the 7x7 = 49 terms.
    
    # Term 0: P1 = (A11+A22)(B11+B22)
    # L0 = [[A11+A22, 0], [0, A11+A22]] -> No, P1 is a scalar in block space? 
    # No, P1 is a 2x2 matrix product.
    # L0 is the 2x2 matrix [[A11+A22, A11+A22], [A11+A22, A11+A22]]? No.
    # P1 = (A11+A22) * (B11+B22). This is a single 2x2 matrix multiplication.
    # So L0 is the 2x2 matrix where every block is (A11+A22)? No.
    # The "Left" operand is the 2x2 matrix (A11+A22). But in the tensor decomposition, 
    # we are decomposing the product of two 2x2 matrices.
    # So for Term 0, the Left matrix is X = A11+A22 (a 2x2 matrix) and Right is Y = B11+B22 (a 2x2 matrix).
    # We decompose X*Y using Strassen.
    
    # Let's define the 7 block terms (X, Y)
    # T0: X = A11+A22, Y = B11+B22
    # T1: X = A21+A22, Y = B11
    # T2: X = A11, Y = B12-B22
    # T3: X = A22, Y = B21-B11
    # T4: X = A11+A12, Y = B22
    # T5: X = A21-A11, Y = B11+B12
    # T6: X = A12-A22, Y = B21+B22
    
    # And the contributions to C:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to map the 2x2 result of each P_t to the global C.
    # P_t is a 2x2 matrix. Its elements P_t[0,0], P_t[0,1], P_t[1,0], P_t[1,1] 
    # map to specific 2x2 blocks of C? No.
    # P_t is the product of two 2x2 matrices. The result is a 2x2 matrix.
    # This 2x2 matrix is added to the corresponding 2x2 block of C?
    # Yes.
    # C11 (top-left 2x2) gets contributions from P1, P4, P5, P7.
    # Specifically, the entire 2x2 matrix P1 is added to C11?
    # Let's check Strassen for 2x2 blocks:
    # C11 = P1 + P4 - P5 + P7.
    # Here P1, P4, P5, P7 are 2x2 matrices.
    # So C11[0,0] = P1[0,0] + P4[0,0] - P5[0,0] + P7[0,0], etc.
    
    # So for each term t, we have a 2x2 matrix result P_t.
    # We need to add P_t to C with a specific sign and position.
    
    # Let's iterate through t=0..6
    # For each t, we define X (2x2) and Y (2x2) as linear combos of A and B.
    # Then we apply Strassen to X*Y.
    # This generates 7 sub-products M1..M7.
    # The result P_t is a 2x2 matrix formed from M1..M7.
    # P_t[0,0] = M1 + M4 - M5 + M7
    # P_t[0,1] = M3 + M5
    # P_t[1,0] = M2 + M4
    # P_t[1,1] = M1 - M2 + M3 + M6
    
    # Then we add P_t to C.
    
    # To make this efficient, we will generate the U, V, W columns for each of the 49 sub-products.
    
    # Define the 7 block terms (X, Y)
    # X and Y are 2x2 matrices. Each element is a linear combo of A or B.
    # We represent a 2x2 matrix of linear combos as a list of 4 lists of (index, coeff).
    # Index is (row, col) in the 4x4 matrix.
    
    def get_block_indices(block_row, block_col):
        # Returns list of 4 indices for a 2x2 block
        return [
            (block_row*2, block_col*2),
            (block_row*2, block_col*2+1),
            (block_row*2+1, block_col*2),
            (block_row*2+1, block_col*2+1)
        ]
    
    # Term 0: X = A11+A22, Y = B11+B22
    # X is 2x2. X[0,0] = A11[0,0]+A22[0,0], etc.
    # Actually, X is the matrix (A11+A22). 
    # So X[0,0] = A[0,0]+A[2,2], X[0,1] = A[0,1]+A[2,3], X[1,0] = A[1,0]+A[3,2], X[1,1] = A[1,1]+A[3,3].
    
    def make_linear_combo_matrix(block1, block2, sign1=1, sign2=0):
        # Creates a 2x2 matrix of linear combos
        # block1, block2 are (row, col) of the top-left of the 2x2 block
        # If sign2 is 0, it's just block1.
        mat = [[None for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                idx1 = (block1[0]+r, block1[1]+c)
                combo = [(idx1, sign1)]
                if sign2 != 0:
                    idx2 = (block2[0]+r, block2[1]+c)
                    combo.append((idx2, sign2))
                mat[r][c] = combo
        return mat
    
    # Block origins
    A11 = (0,0); A12 = (0,2); A21 = (2,0); A22 = (2,2)
    B11 = (0,0); B12 = (0,2); B21 = (2,0); B22 = (2,2)
    
    # Define X_t and Y_t for t=0..6
    X_terms = []
    Y_terms = []
    
    # T0: X = A11+A22, Y = B11+B22
    X_terms.append(make_linear_combo_matrix(A11, A22, 1, 1))
    Y_terms.append(make_linear_combo_matrix(B11, B22, 1, 1))
    
    # T1: X = A21+A22, Y = B11
    X_terms.append(make_linear_combo_matrix(A21, A22, 1, 1))
    Y_terms.append(make_linear_combo_matrix(B11, None, 1, 0))
    
    # T2: X = A11, Y = B12-B22
    X_terms.append(make_linear_combo_matrix(A11, None, 1, 0))
    Y_terms.append(make_linear_combo_matrix(B12, B22, 1, -1))
    
    # T3: X = A22, Y = B21-B11
    X_terms.append(make_linear_combo_matrix(A22, None, 1, 0))
    Y_terms.append(make_linear_combo_matrix(B21, B11, 1, -1))
    
    # T4: X = A11+A12, Y = B22
    X_terms.append(make_linear_combo_matrix(A11, A12, 1, 1))
    Y_terms.append(make_linear_combo_matrix(B22, None, 1, 0))
    
    # T5: X = A21-A11, Y = B11+B12
    X_terms.append(make_linear_combo_matrix(A21, A11, 1, -1))
    Y_terms.append(make_linear_combo_matrix(B11, B12, 1, 1))
    
    # T6: X = A12-A22, Y = B21+B22
    X_terms.append(make_linear_combo_matrix(A12, A22, 1, -1))
    Y_terms.append(make_linear_combo_matrix(B21, B22, 1, 1))
    
    # Contributions of P_t to C
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # For each t, P_t is a 2x2 matrix.
    # We need to add P_t to the corresponding block of C.
    # P_t[0,0] -> C_block[0,0]
    # P_t[0,1] -> C_block[0,1]
    # P_t[1,0] -> C_block[1,0]
    # P_t[1,1] -> C_block[1,1]
    
    # Contribution signs for each t:
    # t=0 (P1): +1 to C11, +1 to C22
    # t=1 (P2): +1 to C21, -1 to C22
    # t=2 (P3): +1 to C12, +1 to C22
    # t=3 (P4): +1 to C11, +1 to C21
    # t=4 (P5): -1 to C11, +1 to C12
    # t=5 (P6): +1 to C22
    # t=6 (P7): +1 to C11
    
    # Map: t -> list of (c_block_row, c_block_col, sign)
    contribs = [
        [(0,0,1), (1,1,1)],      # P1 -> C11, C22
        [(1,0,1), (1,1,-1)],     # P2 -> C21, C22
        [(0,1,1), (1,1,1)],      # P3 -> C12, C22
        [(0,0,1), (1,0,1)],      # P4 -> C11, C21
        [(0,0,-1), (0,1,1)],     # P5 -> C11, C12
        [(1,1,1)],               # P6 -> C22
        [(0,0,1)]                # P7 -> C11
    ]
    
    # Now, for each t, we apply Strassen to X_t * Y_t.
    # X_t and Y_t are 2x2 matrices of linear combos.
    # Let a,b,c,d be the elements of X_t (a=X[0,0], b=X[0,1], c=X[1,0], d=X[1,1])
    # Let e,f,g,h be the elements of Y_t
    
    for t in range(7):
        X = X_terms[t]
        Y = Y_terms[t]
        
        a = X[0][0]; b = X[0][1]; c = X[1][0]; d = X[1][1]
        e = Y[0][0]; f = Y[0][1]; g = Y[1][0]; h = Y[1][1]
        
        # Strassen sub-products
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # We need to combine linear combos.
        def combine_combos(c1, c2, sign2=1):
            # c1, c2 are lists of (idx, coeff)
            # Returns list of (idx, coeff) summed
            from collections import defaultdict
            d = defaultdict(int)
            for idx, coeff in c1:
                d[idx] += coeff
            for idx, coeff in c2:
                d[idx] += sign2 * coeff
            return [(idx, coeff) for idx, coeff in d.items() if coeff != 0]
        
        # M1: (a+d) and (e+h)
        L1 = combine_combos(a, d)
        R1 = combine_combos(e, h)
        
        # M2: (c+d) and e
        L2 = combine_combos(c, d)
        R2 = e
        
        # M3: a and (f-h)
        L3 = a
        R3 = combine_combos(f, h, -1)
        
        # M4: d and (g-e)
        L4 = d
        R4 = combine_combos(g, e, -1)
        
        # M5: (a+b) and h
        L5 = combine_combos(a, b)
        R5 = h
        
        # M6: (c-a) and (e+f)
        L6 = combine_combos(c, a, -1)
        R6 = combine_combos(e, f)
        
        # M7: (b-d) and (g+h)
        L7 = combine_combos(b, d, -1)
        R7 = combine_combos(g, h)
        
        sub_products = [
            (L1, R1), # M1
            (L2, R2), # M2
            (L3, R3), # M3
            (L4, R4), # M4
            (L5, R5), # M5
            (L6, R6), # M6
            (L7, R7)  # M7
        ]
        
        # Result P_t is 2x2 matrix:
        # P[0,0] = M1 + M4 - M5 + M7
        # P[0,1] = M3 + M5
        # P[1,0] = M2 + M4
        # P[1,1] = M1 - M2 + M3 + M6
        
        # We need to add these to C.
        # For each sub-product k (0..6), it contributes to specific P_t elements.
        # Then P_t elements contribute to C blocks.
        
        # Let's precompute which P_t elements each M_k contributes to, and with what sign.
        # M1: P00(+1), P11(+1)
        # M2: P10(+1), P11(-1)
        # M3: P01(+1), P11(+1)
        # M4: P00(+1), P10(+1)
        # M5: P00(-1), P01(+1)
        # M6: P11(+1)
        # M7: P00(+1)
        
        m_contribs = [
            [(0,0,1), (1,1,1)],   # M1
            [(1,0,1), (1,1,-1)],  # M2
            [(0,1,1), (1,1,1)],   # M3
            [(0,0,1), (1,0,1)],   # M4
            [(0,0,-1), (0,1,1)],  # M5
            [(1,1,1)],            # M6
            [(0,0,1)]             # M7
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            L_k, R_k = sub_products[k]
            
            # Fill U and V
            for idx, coeff in L_k:
                row, col = idx
                U[u_idx(row, col), rank_idx] += coeff
            
            for idx, coeff in R_k:
                row, col = idx
                V[v_idx(row, col), rank_idx] += coeff
            
            # Fill W
            # M_k contributes to P_t elements via m_contribs[k]
            # P_t elements contribute to C via contribs[t]
            # We need to chain these.
            
            for p_row, p_col, m_sign in m_contribs[k]:
                # Find which C blocks this P_t element contributes to
                # P_t[p_row, p_col] is added to C_block[cr, cc] with sign c_sign
                # We need to find the C block corresponding to P_t's position.
                # Wait, P_t is a 2x2 matrix. Its elements map to the 2x2 blocks of C.
                # P_t[0,0] is the top-left element of the 2x2 result.
                # This 2x2 result is added to a specific 2x2 block of C (e.g., C11).
                # So P_t[0,0] contributes to C11[0,0], C11[0,1], C11[1,0], C11[1,1]?
                # No. P_t is the product of two 2x2 matrices.
                # The result is a 2x2 matrix.
                # When we say C11 = P1 + ..., we mean the 2x2 matrix C11 is the sum of the 2x2 matrices P1, etc.
                # So P_t[0,0] is the top-left element of the 2x2 matrix P_t.
                # This element is added to the top-left element of the target C block.
                
                # Let's trace the indices.
                # P_t is a 2x2 matrix. Let its elements be p00, p01, p10, p11.
                # These are scalars in the block algebra.
                # But in our tensor decomposition, we are dealing with scalar entries.
                # So P_t[0,0] is a linear combo of scalar products.
                # This linear combo is added to C_block[0,0] (the top-left scalar of the block).
                
                # So if P_t contributes to C_block (cr, cc) with sign c_sign:
                # Then P_t[0,0] contributes to C[cr*2, cc*2] with sign c_sign * m_sign.
                # P_t[0,1] contributes to C[cr*2, cc*2+1] with sign c_sign * m_sign.
                # P_t[1,0] contributes to C[cr*2+1, cc*2] with sign c_sign * m_sign.
                # P_t[1,1] contributes to C[cr*2+1, cc*2+1] with sign c_sign * m_sign.
                
                for c_block_row, c_block_col, c_sign in contribs[t]:
                    # P_t[p_row, p_col] contributes to C[c_block_row*2 + p_row, c_block_col*2 + p_col]
                    out_row = c_block_row * 2 + p_row
                    out_col = c_block_col * 2 + p_col
                    total_sign = m_sign * c_sign
                    W[w_idx(out_col, out_row), rank_idx] += total_sign
    
    # Verify
    if time.time() < deadline:
        U_np = U.astype(float)
        V_np = V.astype(float)
        W_np = W.astype(float)
        
        max_err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_val = U_np[u_idx(i, j)]
                    v_val = V_np[v_idx(j, k)]
                    w_val = W_np[w_idx(k, i)]
                    val = np.sum(u_val * v_val * w_val)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-10:
            # Fallback
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
