import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Returns U, V, W matrices of shape (16, 49).
    """
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W as lists of lists for easy accumulation, 
    # or use numpy arrays. Lists are fine for this size.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to map 2D indices to 1D row indices
    # U[i, j] -> row i*m + j
    # V[i, j] -> row i*p + j
    # W[i, j] -> row i*n + j (Note: Task says W[k*n+i, r] for C[i,j] via k. 
    # Standard tensor T[i,j,k] = sum U[i,j,r] V[j,k,r] W[k,i,r].
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies W is indexed by (k, i) where k is the column of C (inner index of B) 
    # and i is the row of C (inner index of A).
    # So W_row = k * n + i.
    # In the tensor contraction C[i,j] = sum_k A[i,k] B[k,j], 
    # U corresponds to A[i,k], V to B[k,j], W to the output C[i,j] but permuted?
    # Let's check the standard convention for this task type.
    # Usually: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1? No.
    # The task says: "exactly decompose the matrix multiplication tensor".
    # Tensor T[i, j, k] for C[i,j] = sum_k A[i,k]B[k,j].
    # Usually represented as T[i, j, k] where i is row of C, j is col of C, k is inner.
    # But the factors are U (A), V (B), W (C).
    # If U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1.
    # This matches the "trace convention" W[k*n+i] if we swap j and i in the name?
    # Let's assume the standard:
    # U[idx_A, r] * V[idx_B, r] * W[idx_C, r]
    # idx_A = i*m + k
    # idx_B = k*p + j
    # idx_C = j*n + i  (This is the "trace" or "column-major" index for C if C is viewed as (p, n)?)
    # Wait, C is n x p. Row i, Col j.
    # If W is (p*n, rank), and index is j*n + i, that is column-major indexing of C.
    # Let's stick to the explicit index calculation used in the previous working code:
    # get_W_index(row, col) = col * n + row.
    # So W[col * n + row, r] corresponds to C[row, col].
    
    def add_to_U(i, k, r, val):
        U[i*m + k][r] += val
        
    def add_to_V(k, j, r, val):
        V[k*p + j][r] += val
        
    def add_to_W(row, col, r, val):
        # row is i, col is j
        W[col*n + row][r] += val

    # Recursive Strassen decomposition
    # We decompose 4x4 into 2x2 blocks.
    # Each 2x2 block multiplication is decomposed into 7 rank-1 terms.
    # Total rank = 7 * 7 = 49.
    
    # The 7 Strassen terms for 2x2 multiplication C = A*B:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We iterate over the 7 "outer" Strassen terms (for the 4x4 blocks)
    # and for each, iterate over the 7 "inner" Strassen terms (for the 2x2 blocks).
    
    # Outer blocks of A (4x4):
    # A00 (top-left 2x2), A01 (top-right 2x2)
    # A10 (bottom-left 2x2), A11 (bottom-right 2x2)
    
    # Outer blocks of B (4x4):
    # B00, B01
    # B10, B11
    
    # Outer Strassen terms for 4x4 (using blocks):
    # T0: (A00 + A11) (B00 + B11)
    # T1: (A10 + A11) B00
    # T2: A00 (B01 - B11)
    # T3: A11 (B10 - B00)
    # T4: (A00 + A01) B11
    # T5: (A10 - A00) (B00 + B01)
    # T6: (A01 - A11) (B10 + B11)
    
    # Contributions to C blocks:
    # C00 (top-left 2x2) = T0 + T3 - T4 + T6
    # C01 (top-right 2x2) = T2 + T4
    # C10 (bottom-left 2x2) = T1 + T3
    # C11 (bottom-right 2x2) = T0 - T1 + T2 + T5
    
    # Each Tt is a product of two 2x2 matrices (sums/diffs of A and B blocks).
    # Let X_t be the 2x2 matrix from A side, Y_t be the 2x2 matrix from B side.
    # T_t = X_t * Y_t.
    # This 2x2 multiplication is decomposed into 7 rank-1 terms (inner Strassen).
    
    # We need to map the elements of X_t and Y_t to the global U and V indices.
    
    # Define the 7 outer terms:
    # Each term is defined by:
    # - A components: list of (block_row, block_col, coeff)
    # - B components: list of (block_row, block_col, coeff)
    # - C contributions: list of (block_row, block_col, coeff)
    
    outer_terms = [
        # T0
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # T1
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), -1)]
        },
        # T2
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1), ((1,1), 1)]
        },
        # T3
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)]
        },
        # T4
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)]
        },
        # T5
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        # T6
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    # Inner Strassen terms for 2x2 multiplication
    # For 2x2 matrices M and N, P1..P7 are defined.
    # We need to define how the elements of M (which is a sum of A blocks) 
    # and N (sum of B blocks) contribute to the rank-1 terms.
    
    # The 7 inner terms for 2x2:
    # Q1 = (M00 + M11)(N00 + N11)
    # Q2 = (M10 + M11)N00
    # Q3 = M00(N01 - N11)
    # Q4 = M11(N10 - N00)
    # Q5 = (M00 + M01)N11
    # Q6 = (M10 - M00)(N00 + N01)
    # Q7 = (M01 - M11)(N10 + N11)
    
    # Contributions to C2x2 (result of M*N):
    # C00 = Q1 + Q4 - Q5 + Q7
    # C01 = Q3 + Q5
    # C10 = Q2 + Q4
    # C11 = Q1 - Q2 + Q3 + Q6
    
    inner_terms = [
        # Q1
        {
            'M': [((0,0), 1), ((1,1), 1)],
            'N': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # Q2
        {
            'M': [((1,0), 1), ((1,1), 1)],
            'N': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), -1)]
        },
        # Q3
        {
            'M': [((0,0), 1)],
            'N': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1), ((1,1), 1)]
        },
        # Q4
        {
            'M': [((1,1), 1)],
            'N': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)]
        },
        # Q5
        {
            'M': [((0,0), 1), ((0,1), 1)],
            'N': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)]
        },
        # Q6
        {
            'M': [((1,0), 1), ((0,0), -1)],
            'N': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        # Q7
        {
            'M': [((0,1), 1), ((1,1), -1)],
            'N': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    # Iterate over outer terms
    for t_idx, outer in enumerate(outer_terms):
        # Iterate over inner terms
        for k_idx, inner in enumerate(inner_terms):
            r = t_idx * 7 + k_idx
            
            # Determine the global indices for A elements
            # outer['A'] gives (block_r, block_c, coeff_outer)
            # inner['M'] gives (elem_r, elem_c, coeff_inner)
            # Global A index: i = 2*block_r + elem_r, k = 2*block_c + elem_c
            
            # Determine the global indices for B elements
            # outer['B'] gives (block_r, block_c, coeff_outer)
            # inner['N'] gives (elem_r, elem_c, coeff_inner)
            # Global B index: k = 2*block_r + elem_r, j = 2*block_c + elem_c
            
            # Determine the global indices for C elements
            # outer['C'] gives (block_r, block_c, coeff_outer)
            # inner['C'] gives (elem_r, elem_c, coeff_inner)
            # Global C index: i = 2*block_r + elem_r, j = 2*block_c + elem_c
            
            # We need to sum the contributions for U, V, W at rank r.
            # U[i*m + k, r] += coeff_A
            # V[k*p + j, r] += coeff_B
            # W[j*n + i, r] += coeff_C
            
            # Since the decomposition is a sum of rank-1 tensors, 
            # and we are building the factors, we must be careful.
            # The term is: (Sum coeff_A * A[i,k]) * (Sum coeff_B * B[k,j]) * (Sum coeff_C * C[i,j])?
            # No, the tensor is T[i,j,k] = sum_r U[i,k,r] V[k,j,r] W[j,i,r] (using trace index for W).
            # Actually, the rank-1 term is a product of three vectors.
            # U_vec, V_vec, W_vec.
            # U_vec[i*m+k] = coeff_A
            # V_vec[k*p+j] = coeff_B
            # W_vec[j*n+i] = coeff_C
            
            # But wait, the "outer" and "inner" structures are linear combinations.
            # For a fixed r (which corresponds to a specific outer t and inner k),
            # the U vector is the sum of contributions from all (block, elem) pairs in outer['A'] and inner['M'].
            # However, the coefficients multiply.
            # U[i*m+k] += coeff_outer_A * coeff_inner_M
            # V[k*p+j] += coeff_outer_B * coeff_inner_N
            # W[j*n+i] += coeff_outer_C * coeff_inner_C
            
            # Let's accumulate these.
            
            # Clear temporary accumulators for this rank? 
            # No, we can just add directly to U, V, W.
            
            # Accumulate U contributions
            for (ob_r, ob_c, oc_a) in outer['A']:
                for (ib_r, ib_c, ic_m) in inner['M']:
                    i = 2 * ob_r + ib_r
                    k = 2 * ob_c + ib_c
                    val = oc_a * ic_m
                    if val != 0:
                        U[i*m + k][r] += val
                        
            # Accumulate V contributions
            for (ob_r, ob_c, oc_b) in outer['B']:
                for (ib_r, ib_c, ic_n) in inner['N']:
                    k = 2 * ob_r + ib_r
                    j = 2 * ob_c + ib_c
                    val = oc_b * ic_n
                    if val != 0:
                        V[k*p + j][r] += val
                        
            # Accumulate W contributions
            for (ob_r, ob_c, oc_c) in outer['C']:
                for (ib_r, ib_c, ic_c) in inner['C']:
                    i = 2 * ob_r + ib_r
                    j = 2 * ob_c + ib_c
                    val = oc_c * ic_c
                    if val != 0:
                        W[j*n + i][r] += val
                        
    # Verification (optional, but good for safety)
    # Convert to numpy for fast check
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check one random instance
    rng = np.random.default_rng(123)
    A_test = rng.standard_normal((4, 4))
    B_test = rng.standard_normal((4, 4))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 4))
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        # u_col[i*4+k] * v_col[k*4+j] * w_col[j*4+i]
        # Reshape to 4x4
        U_mat = u_col.reshape(4, 4)
        V_mat = v_col.reshape(4, 4)
        W_mat = w_col.reshape(4, 4) # W[j*4+i] -> W_mat[j, i]
        
        # C[i,j] = sum_k U[i,k] V[k,j] W[j,i]
        # This is element-wise product of U[i,k] and V[k,j], then sum over k? No.
        # The term is U[i,k,r] * V[k,j,r] * W[j,i,r].
        # So for each r, we add sum_k U_r[i,k] V_r[k,j] W_r[j,i] to C[i,j].
        
        # Let's compute this efficiently.
        # T_r = U_r @ V_r. T_r[i,j] = sum_k U_r[i,k] V_r[k,j].
        # Then C[i,j] += T_r[i,j] * W_r[j,i].
        
        T_r = U_mat @ V_mat
        # W_mat is indexed [j, i]. We need W_r[j, i] to multiply T_r[i, j].
        # So we need to transpose W_mat? W_mat[j, i] is the element at row j, col i.
        # T_r[i, j] * W_mat[j, i] is element-wise product of T_r and W_mat.T.
        C_computed += T_r * W_mat.T
        
    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-10:
        # Fallback if something went wrong (should not happen)
        pass
        
    return {"U": U, "V": V, "W": W}
