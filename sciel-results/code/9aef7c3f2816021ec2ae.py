import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # We will construct a rank 63 solution using Strassen on 2x2 blocks of 3x3 matrices
    # embedded in 6x6 matrices.
    
    # Strassen's 7 multiplications for 2x2 block matrix
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a product of two 3x3 matrices.
    # We use schoolbook multiplication for 3x3 (rank 9).
    # Total rank = 7 * 9 = 63.
    
    R = 63
    U = np.zeros((36, R), dtype=int) # 6x6 matrix A
    V = np.zeros((36, R), dtype=int) # 6x6 matrix B
    W = np.zeros((36, R), dtype=int) # 6x6 matrix C
    
    # Helper to add a schoolbook 3x3 multiplication to the decomposition
    # a_start, a_end: indices in A (0..5)
    # b_start, b_end: indices in B (0..5)
    # c_start, c_end: indices in C (0..5)
    # signs_a, signs_b: coefficients for A and B blocks
    # sign_c: coefficient for C block
    def add_schoolbook_3x3(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, coeff_a, coeff_b, coeff_c, offset):
        # a_rows, a_cols are tuples of row/col indices in the 6x6 matrix
        # We need to map these to the 3x3 block structure.
        # Actually, it's easier to just iterate over the 9 entries of the 3x3 block.
        pass

    # It's easier to construct the 3x3 schoolbook decomposition explicitly.
    # U_3x3, V_3x3, W_3x3 are 9x9 matrices (rank 9).
    # U_3x3[i*3+j, k] = 1 if k == i*3+j
    # V_3x3[i*3+j, k] = 1 if k == i*3+j
    # W_3x3[i*3+j, k] = 1 if k == i*3+j
    # This is just the identity.
    
    # Let's define the 7 Strassen terms.
    # Each term involves linear combinations of A and B blocks.
    # We will create 7 sets of 9 rank-1 tensors.
    
    # Term 1: (A11+A22)(B11+B22)
    # A_part = A11 + A22. B_part = B11 + B22.
    # C_part = M1.
    
    # We need to express A11, A22 etc in terms of the full 6x6 A.
    # A11 is rows 0-2, cols 0-2.
    # A12 is rows 0-2, cols 3-5.
    # A21 is rows 3-5, cols 0-2.
    # A22 is rows 3-5, cols 3-5.
    
    # For each of the 7 terms, we have a linear combination of A blocks and B blocks.
    # Let A_comb = c11*A11 + c12*A12 + c21*A21 + c22*A22
    # Let B_comb = d11*B11 + d12*B12 + d21*B21 + d22*B22
    # Then M = A_comb * B_comb.
    # We add this to the decomposition.
    
    # The rank-1 tensors for A_comb * B_comb are formed by the tensor product of the decompositions of A_comb and B_comb?
    # No, A_comb and B_comb are matrices.
    # We multiply them using schoolbook (9 mults).
    # So for each of the 9 schoolbook multiplications, we have:
    # u_vec = e_i (row of A_comb)
    # v_vec = e_j (col of B_comb)
    # w_vec = e_k (entry of C)
    
    # So we need to build the vectors for A_comb and B_comb.
    
    # Let's define the 7 terms with their coefficients.
    # (c11, c12, c21, c22) for A, (d11, d12, d21, d22) for B, and (w11, w12, w21, w22) for C contribution.
    # C = [[C11, C12], [C21, C22]]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: (c11, c12, c21, c22)
        # B_coeffs: (d11, d12, d21, d22)
        # C_coeffs: (w11, w12, w21, w22)
        ({1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}),   # M1: (A11+A22)(B11+B22) -> C11 += 1
        ({0, 0, 1, 1}, {1, 0, 0, 0}, {0, 0, 1, 0}),   # M2: (A21+A22)B11 -> C21 += 1
        ({1, 0, 0, 0}, {0, 1, 0, -1}, {0, 1, 0, 0}),  # M3: A11(B12-B22) -> C12 += 1
        ({0, 0, 0, 1}, {0, 0, 1, -1}, {1, 0, 0, 0}),  # M4: A22(B21-B11) -> C11 += 1
        ({1, 1, 0, 0}, {0, 0, 0, 1}, {-1, 1, 0, 0}),  # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
        ({-1, 0, 1, 0}, {1, 1, 0, 0}, {0, 0, 0, 1}),  # M6: (A21-A11)(B11+B12) -> C22 += 1
        ({0, 1, 0, -1}, {0, 0, 1, 1}, {1, 0, 0, 1}),  # M7: (A12-A22)(B21+B22) -> C11 += 1, C22 += 1
    ]
    
    # Wait, the C coefficients in the tuple above are for the 2x2 block structure of C.
    # C11, C12, C21, C22 are 3x3 matrices.
    # So for each term, we add the contribution to the specific 3x3 block of C.
    
    # We will fill U, V, W for the 6x6 case.
    # Index in U: i*6 + j.
    # Index in V: i*6 + j.
    # Index in W: i*6 + j.
    
    col = 0
    for a_coeffs, b_coeffs, c_coeffs in terms:
        # a_coeffs: dict or tuple? Let's use tuples (c11, c12, c21, c22)
        # I used dicts in the list above, let's fix that.
        pass
        
    # Let's restart the term definition with tuples.
    terms = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 0)),   # M1 -> C11
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),   # M2 -> C21
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)),  # M3 -> C12
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 0, 0)),  # M4 -> C11
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),  # M5 -> C11, C12
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),  # M6 -> C22
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 1)),  # M7 -> C11, C22
    ]
    
    # Note: The last tuple in each term represents the contribution to C blocks.
    # (w11, w12, w21, w22)
    
    # For each term, we perform 9 schoolbook multiplications.
    for a_c, b_c, c_c in terms:
        # a_c: (c11, c12, c21, c22)
        # b_c: (d11, d12, d21, d22)
        # c_c: (w11, w12, w21, w22)
        
        # We need to form the vectors for A_comb and B_comb.
        # A_comb is a 6x6 matrix.
        # A_comb[i,j] = sum of a_c[k] * A_block[k][i,j]
        
        # We will iterate over the 9 entries of the 3x3 result.
        # Result entry (r, s) where r,s in 0..2.
        # This entry is computed as sum_{t=0..2} A_comb[i, t] * B_comb[t, j]
        # where i, j are the global row/col indices in the 6x6 matrix.
        
        # Actually, it's simpler to just build the U, V, W vectors directly.
        # For each of the 9 schoolbook multiplications (indexed by r, s, t in 0..2):
        # u_vec has non-zeros in the A blocks.
        # v_vec has non-zeros in the B blocks.
        # w_vec has non-zeros in the C blocks.
        
        for r in range(3):
            for s in range(3):
                for t in range(3):
                    # This corresponds to C[r,s] += A[r,t] * B[t,s]
                    # In the 6x6 context:
                    # A_row = r (if in top block) or r+3 (if in bottom)
                    # A_col = t (if in left block) or t+3 (if in right)
                    
                    # We need to determine which A blocks contribute to A[r_global, t_global].
                    # The term is a linear combination of A blocks.
                    # A_comb = c11 A11 + c12 A12 + c21 A21 + c22 A22
                    
                    # Let's build the u_vec, v_vec, w_vec for this specific multiplication.
                    u_vec = np.zeros(36)
                    v_vec = np.zeros(36)
                    w_vec = np.zeros(36)
                    
                    # A part:
                    # A[r_global, t_global]
                    # The global row r_global depends on which block of C we are computing?
                    # No, the term M is a 3x3 matrix.
                    # M is added to specific C blocks.
                    # The row index of M is 0..2.
                    # If M is added to C11 (top-left), the global row is r.
                    # If M is added to C12 (top-right), the global row is r.
                    # If M is added to C21 (bottom-left), the global row is r+3.
                    # If M is added to C22 (bottom-right), the global row is r+3.
                    
                    # Similarly for columns.
                    
                    # Let's determine the C block(s) this term contributes to.
                    # c_c = (w11, w12, w21, w22)
                    # We need to add the contribution to each block where w != 0.
                    
                    # However, the u_vec and v_vec are the SAME for all C blocks in this term.
                    # Only w_vec changes.
                    
                    # So we build u_vec and v_vec once for the term (r, s, t).
                    
                    # u_vec:
                    # A_comb[r_global, t_global]
                    # r_global, t_global are determined by the block position in A.
                    # A11: rows 0-2, cols 0-2.
                    # A12: rows 0-2, cols 3-5.
                    # A21: rows 3-5, cols 0-2.
                    # A22: rows 3-5, cols 3-5.
                    
                    # The "local" row is r, local col is t.
                    # If A11 is involved, global row is r, global col is t.
                    # If A12 is involved, global row is r, global col is t+3.
                    # If A21 is involved, global row is r+3, global col is t.
                    # If A22 is involved, global row is r+3, global col is t+3.
                    
                    # Coefficients for A blocks: a_c = (c11, c12, c21, c22)
                    # c11 corresponds to A11.
                    # c12 corresponds to A12.
                    # c21 corresponds to A21.
                    # c22 corresponds to A22.
                    
                    if a_c[0] != 0: # A11
                        u_vec[r * 6 + t] += a_c[0]
                    if a_c[1] != 0: # A12
                        u_vec[r * 6 + (t + 3)] += a_c[1]
                    if a_c[2] != 0: # A21
                        u_vec[(r + 3) * 6 + t] += a_c[2]
                    if a_c[3] != 0: # A22
                        u_vec[(r + 3) * 6 + (t + 3)] += a_c[3]
                    
                    # v_vec:
                    # B_comb[t_global, s_global]
                    # B11: rows 0-2, cols 0-2.
                    # B12: rows 0-2, cols 3-5.
                    # B21: rows 3-5, cols 0-2.
                    # B22: rows 3-5, cols 3-5.
                    
                    # Local row is t, local col is s.
                    # Coefficients for B blocks: b_c = (d11, d12, d21, d22)
                    
                    if b_c[0] != 0: # B11
                        v_vec[t * 6 + s] += b_c[0]
                    if b_c[1] != 0: # B12
                        v_vec[t * 6 + (s + 3)] += b_c[1]
                    if b_c[2] != 0: # B21
                        v_vec[(t + 3) * 6 + s] += b_c[2]
                    if b_c[3] != 0: # B22
                        v_vec[(t + 3) * 6 + (s + 3)] += b_c[3]
                    
                    # Now, for each C block, we add the contribution.
                    # w_vec will be different for each C block.
                    # But we are adding rank-1 tensors.
                    # We can add 1 tensor for each C block.
                    
                    if c_c[0] != 0: # C11
                        w_vec = np.zeros(36)
                        w_vec[r * 6 + s] = c_c[0]
                        U[:, col] = u_vec
                        V[:, col] = v_vec
                        W[:, col] = w_vec
                        col += 1
                    if c_c[1] != 0: # C12
                        w_vec = np.zeros(36)
                        w_vec[r * 6 + (s + 3)] = c_c[1]
                        U[:, col] = u_vec
                        V[:, col] = v_vec
                        W[:, col] = w_vec
                        col += 1
                    if c_c[2] != 0: # C21
                        w_vec = np.zeros(36)
                        w_vec[(r + 3) * 6 + s] = c_c[2]
                        U[:, col] = u_vec
                        V[:, col] = v_vec
                        W[:, col] = w_vec
                        col += 1
                    if c_c[3] != 0: # C22
                        w_vec = np.zeros(36)
                        w_vec[(r + 3) * 6 + (s + 3)] = c_c[3]
                        U[:, col] = u_vec
                        V[:, col] = v_vec
                        W[:, col] = w_vec
                        col += 1
    
    # We have filled U, V, W for the 6x6 case.
    # col should be 63.
    
    # Now extract the submatrices for 3x5x6.
    # A is 3x5. Rows 0-2, Cols 0-4.
    # B is 5x6. Rows 0-4, Cols 0-5.
    # C is 3x6. Rows 0-2, Cols 0-5.
    
    # U indices: i*6 + j.
    # We need rows for i in 0..2, j in 0..4.
    U_indices = [i * 6 + j for i in range(3) for j in range(5)]
    U_final = U[U_indices, :]
    
    # V indices: i*6 + j.
    # We need rows for i in 0..4, j in 0..5.
    V_indices = [i * 6 + j for i in range(5) for j in range(6)]
    V_final = V[V_indices, :]
    
    # W indices: i*6 + j.
    # We need rows for i in 0..2, j in 0..5.
    W_indices = [i * 6 + j for i in range(3) for j in range(6)]
    W_final = W[W_indices, :]
    
    # Convert to lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
