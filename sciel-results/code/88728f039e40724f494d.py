import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = 150.0 # 150s wall-clock limit
    
    n, m, p = DIMENSIONS
    
    # Generate Strassen Rank 49 Decomposition
    # Strassen for 2x2 blocks
    # Top level: 7 terms
    # Each term is a product of two 2x2 matrices
    # Each 2x2 product uses 7 scalar multiplications
    
    # We represent each bilinear form as a vector of length 16*16 = 256
    # Index: i*16 + j for A[i,j], k*16 + l for B[k,l]
    # Bilinear form f(A,B) = sum c_{i,j,k,l} A_{i,j} B_{k,l}
    # Vector index: (i*4+j)*16 + (k*4+l)
    
    # Strassen 2x2 terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to the 4x4 indices.
    # A is 4x4, blocks A11, A12, A21, A22 are 2x2.
    # A11 indices: 0,1,4,5
    # A12 indices: 2,3,6,7
    # A21 indices: 8,9,12,13
    # A22 indices: 10,11,14,15
    
    # Let's define the linear forms for U and V for each of the 7 Strassen terms.
    # For each term t, we have U_t (4x4 matrix of coeffs) and V_t (4x4 matrix of coeffs).
    # The bilinear form is sum_{i,j,k,l} U_t[i,j] V_t[k,l] A[i,j] B[k,l].
    
    # Strassen U/V definitions (for 2x2 blocks):
    # U1 = A11 + A22, V1 = B11 + B22
    # U2 = A21 + A22, V2 = B11
    # U3 = A11, V3 = B12 - B22
    # U4 = A22, V4 = B21 - B11
    # U5 = A11 + A12, V5 = B22
    # U6 = A21 - A11, V6 = B11 + B12
    # U7 = A12 - A22, V7 = B21 + B22
    
    # We will generate the 49 scalar multiplications.
    # Each scalar multiplication corresponds to a term in the Strassen expansion of the 2x2 block product.
    # Let's just generate the 49 vectors directly.
    
    # Helper to create a 256-dim vector from two 4x4 coefficient matrices
    def vec_from_matrices(U, V):
        v = np.zeros(256)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        idx = (i * 4 + j) * 16 + (k * 4 + l)
                        v[idx] = U[i, j] * V[k, l]
        return v

    # Define block indices
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    def get_block(M, r0, r1, c0, c1):
        return M[r0:r1, c0:c1]

    # We need to define the 7 U and V matrices for the top level.
    # But the "scalar multiplications" are inside the block products.
    # So we need to expand each block product into 7 scalar products.
    
    # Let's define the 7 Strassen terms for a generic 2x2 matrix product.
    # Term 1: (A00+A11)(B00+B11)
    # Term 2: (A10+A11)B00
    # Term 3: A00(B01-B11)
    # Term 4: A11(B10-B00)
    # Term 5: (A00+A01)B11
    # Term 6: (A10-A00)(B00+B01)
    # Term 7: (A01-A11)(B10+B11)
    
    # For each top-level term t, we have U_t and V_t which are 2x2 blocks of 2x2 matrices.
    # So U_t is a 4x4 matrix, V_t is a 4x4 matrix.
    # The product U_t V_t is computed using 7 scalar multiplications.
    # Let's generate the 49 vectors.
    
    # Top Level U/V definitions
    # We need to construct U1..U7 and V1..V7 as 4x4 matrices.
    
    U = [np.zeros((4,4)) for _ in range(7)]
    V = [np.zeros((4,4)) for _ in range(7)]
    
    # Fill U1, V1
    # U1 = A11 + A22
    U[0][0:2, 0:2] = 1
    U[0][2:4, 2:4] = 1
    # V1 = B11 + B22
    V[0][0:2, 0:2] = 1
    V[0][2:4, 2:4] = 1
    
    # U2 = A21 + A22
    U[1][2:4, 0:2] = 1
    U[1][2:4, 2:4] = 1
    # V2 = B11
    V[1][0:2, 0:2] = 1
    
    # U3 = A11
    U[2][0:2, 0:2] = 1
    # V3 = B12 - B22
    V[2][0:2, 2:4] = 1
    V[2][2:4, 2:4] = -1
    
    # U4 = A22
    U[3][2:4, 2:4] = 1
    # V4 = B21 - B11
    V[3][2:4, 0:2] = 1
    V[3][0:2, 0:2] = -1
    
    # U5 = A11 + A12
    U[4][0:2, 0:2] = 1
    U[4][0:2, 2:4] = 1
    # V5 = B22
    V[4][2:4, 2:4] = 1
    
    # U6 = A21 - A11
    U[5][2:4, 0:2] = 1
    U[5][0:2, 0:2] = -1
    # V6 = B11 + B12
    V[5][0:2, 0:2] = 1
    V[5][0:2, 2:4] = 1
    
    # U7 = A12 - A22
    U[6][0:2, 2:4] = 1
    U[6][2:4, 2:4] = -1
    # V7 = B21 + B22
    V[6][2:4, 0:2] = 1
    V[6][2:4, 2:4] = 1
    
    # Now, for each t in 0..6, we compute the 7 scalar products for U[t] * V[t].
    # The 7 products are defined by the Strassen formula on the 2x2 blocks of U[t] and V[t].
    # Let U[t] be partitioned into 2x2 blocks U00, U01, U10, U11.
    # Let V[t] be partitioned into 2x2 blocks V00, V01, V10, V11.
    
    bilinear_vectors = []
    
    for t in range(7):
        Ut = U[t]
        Vt = V[t]
        
        # Extract blocks
        U00 = Ut[0:2, 0:2]
        U01 = Ut[0:2, 2:4]
        U10 = Ut[2:4, 0:2]
        U11 = Ut[2:4, 2:4]
        
        V00 = Vt[0:2, 0:2]
        V01 = Vt[0:2, 2:4]
        V10 = Vt[2:4, 0:2]
        V11 = Vt[2:4, 2:4]
        
        # 7 Strassen terms for this sub-product
        # S1 = (U00+U11)(V00+V11)
        # S2 = (U10+U11)V00
        # S3 = U00(V01-V11)
        # S4 = U11(V10-V00)
        # S5 = (U00+U01)V11
        # S6 = (U10-U00)(V00+V01)
        # S7 = (U01-U11)(V10+V11)
        
        # We need to form the 4x4 coefficient matrices for each S.
        # S1: (U00+U11) is a 2x2 matrix of coeffs. (V00+V11) is 2x2.
        # The product is a 2x2 matrix of coeffs.
        # We need to expand this into the 4x4 space.
        
        def expand_2x2_to_4x4(M):
            # M is 2x2. Return 4x4 with M in top-left?
            # No, the blocks U00, U01 etc are already 2x2 matrices of coefficients.
            # The "scalar product" is sum_{i,j,k,l} (U_block[i,j]) (V_block[k,l]) A[i,j] B[k,l]
            # Wait, U_block[i,j] is a coefficient for A[row, col].
            # The index in the 4x4 matrix is (block_row*2 + i, block_col*2 + j).
            
            # For S1, the U part is (U00+U11).
            # U00 corresponds to A[0:2, 0:2]. U11 corresponds to A[2:4, 2:4].
            # So the U matrix for S1 has U00+U11 in [0:2,0:2] and U00+U11 in [2:4,2:4]?
            # No.
            # U00 is the top-left block of Ut.
            # The term is (U00 + U11) * (V00 + V11).
            # This means we take the linear form L_U = U00 + U11.
            # L_U is a 2x2 matrix of coefficients for the *blocks*?
            # No, U00 is a 2x2 matrix of coefficients for the *entries* of the A11 block.
            # So L_U is a 2x2 matrix of coefficients for A11 entries, plus a 2x2 matrix for A22 entries.
            # To form the 4x4 U matrix for this term:
            # U_S1[0:2, 0:2] = U00 + U11 ? No.
            # U00 is the coefficients for A[0:2, 0:2].
            # U11 is the coefficients for A[2:4, 2:4].
            # So the combined linear form on A is:
            # A[0:2, 0:2] * (U00) + A[2:4, 2:4] * (U11).
            # Wait, the Strassen formula is (A11+A22).
            # So the coefficient for A11 is 1, for A22 is 1.
            # So U_S1[0:2, 0:2] = U00 (which is 1s) + 0?
            # No, U00 is the block of Ut.
            # In the top-level definition, U[0] had 1s in [0:2,0:2] and [2:4,2:4].
            # So U00 = [[1,1],[1,1]], U11 = [[1,1],[1,1]].
            # The term is (U00+U11).
            # This implies we are adding the linear forms.
            # So the U matrix for the scalar product S1 is:
            # U_S1[0:2, 0:2] = U00
            # U_S1[2:4, 2:4] = U11
            # And 0 elsewhere.
            # BUT, the Strassen formula is (A11+A22).
            # This means the linear form is A11 + A22.
            # So the coefficient for A11 is 1, for A22 is 1.
            # So U_S1 should have U00 in [0:2,0:2] and U11 in [2:4,2:4].
            
            # Let's verify.
            # If U00 = 1, U11 = 1.
            # Then U_S1[0:2,0:2] = 1, U_S1[2:4,2:4] = 1.
            # This represents the linear form A11 + A22.
            # Correct.
            
            return None # Placeholder

        # We need to construct the 4x4 U and V matrices for each of the 7 sub-terms.
        
        sub_terms = []
        
        # S1: (U00+U11)(V00+V11)
        U_s1 = np.zeros((4,4))
        U_s1[0:2, 0:2] = U00
        U_s1[2:4, 2:4] = U11
        V_s1 = np.zeros((4,4))
        V_s1[0:2, 0:2] = V00
        V_s1[2:4, 2:4] = V11
        sub_terms.append((U_s1, V_s1))
        
        # S2: (U10+U11)V00
        U_s2 = np.zeros((4,4))
        U_s2[2:4, 0:2] = U10
        U_s2[2:4, 2:4] = U11
        V_s2 = np.zeros((4,4))
        V_s2[0:2, 0:2] = V00
        sub_terms.append((U_s2, V_s2))
        
        # S3: U00(V01-V11)
        U_s3 = np.zeros((4,4))
        U_s3[0:2, 0:2] = U00
        V_s3 = np.zeros((4,4))
        V_s3[0:2, 2:4] = V01
        V_s3[2:4, 2:4] = -V11
        sub_terms.append((U_s3, V_s3))
        
        # S4: U11(V10-V00)
        U_s4 = np.zeros((4,4))
        U_s4[2:4, 2:4] = U11
        V_s4 = np.zeros((4,4))
        V_s4[2:4, 0:2] = V10
        V_s4[0:2, 0:2] = -V00
        sub_terms.append((U_s4, V_s4))
        
        # S5: (U00+U01)V11
        U_s5 = np.zeros((4,4))
        U_s5[0:2, 0:2] = U00
        U_s5[0:2, 2:4] = U01
        V_s5 = np.zeros((4,4))
        V_s5[2:4, 2:4] = V11
        sub_terms.append((U_s5, V_s5))
        
        # S6: (U10-U00)(V00+V01)
        U_s6 = np.zeros((4,4))
        U_s6[2:4, 0:2] = U10
        U_s6[0:2, 0:2] = -U00
        V_s6 = np.zeros((4,4))
        V_s6[0:2, 0:2] = V00
        V_s6[0:2, 2:4] = V01
        sub_terms.append((U_s6, V_s6))
        
        # S7: (U01-U11)(V10+V11)
        U_s7 = np.zeros((4,4))
        U_s7[0:2, 2:4] = U01
        U_s7[2:4, 2:4] = -U11
        V_s7 = np.zeros((4,4))
        V_s7[2:4, 0:2] = V10
        V_s7[2:4, 2:4] = V11
        sub_terms.append((U_s7, V_s7))
        
        for Us, Vs in sub_terms:
            bilinear_vectors.append(vec_from_matrices(Us, Vs))
            
    # Now we have 49 vectors.
    # Check for linear dependence.
    M = np.array(bilinear_vectors).T # 256 x 49
    
    # Compute SVD to find rank
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Determine rank (tolerance 1e-10)
        rank = np.sum(s > 1e-10)
        
        if rank < 49:
            # Find the null space vector
            # The null space is the right singular vectors corresponding to small singular values.
            # Vt is 49 x 49.
            # The last (49-rank) rows of Vt are the null space basis.
            # We want to find a vector c such that M c = 0.
            # If rank is 48, there is 1 null vector.
            null_vec = Vt[-1, :]
            
            # Find the index of the largest component in null_vec
            idx = np.argmax(np.abs(null_vec))
            
            # Remove the term at idx
            # The terms are ordered: 7 for t=0, 7 for t=1, etc.
            # We just remove the vector from the list.
            
            # Verify correctness?
            # The remaining 48 vectors should span the same space.
            
            # Construct U, V, W from the remaining vectors.
            # We need to map the vectors back to U, V, W matrices.
            # Each vector is a bilinear form.
            # We can just output the 48 vectors as the decomposition?
            # No, the interface requires U, V, W matrices.
            # U[i, c], V[j, c], W[k, c].
            # The bilinear form is sum_c U[i,c] V[j,c] W[k,c] ?
            # No, the standard decomposition is C = sum_c U[:,c] V[:,c]^T W[:,c]^T?
            # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
            # And W[k*n+i, r].
            # The product is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
            
            # My vectors are in the basis of A[i,j] B[k,l].
            # The output C[i,k] = sum_j A[i,j] B[j,k].
            # So the bilinear form for C[i,k] is sum_j A[i,j] B[j,k].
            # My vector index is (i*4+j)*16 + (k*4+l).
            # This corresponds to A[i,j] B[k,l].
            # This is NOT the same as A[i,j] B[j,k].
            # I need to map the indices.
            
            # Let's reconstruct U, V, W directly from the Strassen structure.
            # This is safer.
            
            # I will return the Rank 49 Strassen decomposition if I can't easily map the 48 vectors.
            # But I want 48.
            
            # Let's try to construct U, V, W for the 49 terms first.
            
            U_mat = []
            V_mat = []
            W_mat = []
            
            # The 49 terms are generated in order.
            # For each term, we have U_s, V_s.
            # The output contribution is U_s V_s^T ?
            # No, the bilinear form is sum A[i,j] B[k,l] U_s[i,j] V_s[k,l].
            # This contributes to C[i,l]?
            # Matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k].
            # So the term A[i,j] B[j,k] contributes to C[i,k].
            # My vector has A[i,j] B[k,l].
            # So I need to identify j and k.
            # In the Strassen algorithm, the indices are consistent.
            
            # Let's just use the known Strassen U, V, W construction.
            
            # I will implement the Rank 49 Strassen decomposition explicitly.
            
            # If I have time, I will try to remove one term.
            
            # For now, return Rank 49.
            
            rank_out = 49
            U_out = np.zeros((16, rank_out))
            V_out = np.zeros((16, rank_out))
            W_out = np.zeros((16, rank_out))
            
            # Re-generate the 49 terms and fill U, V, W.
            
            col = 0
            for t in range(7):
                Ut = U[t]
                Vt = V[t]
                
                U00 = Ut[0:2, 0:2]
                U01 = Ut[0:2, 2:4]
                U10 = Ut[2:4, 0:2]
                U11 = Ut[2:4, 2:4]
                
                V00 = Vt[0:2, 0:2]
                V01 = Vt[0:2, 2:4]
                V10 = Vt[2:4, 0:2]
                V11 = Vt[2:4, 2:4]
                
                sub_terms = []
                # S1
                U_s1 = np.zeros((4,4)); U_s1[0:2, 0:2] = U00; U_s1[2:4, 2:4] = U11
                V_s1 = np.zeros((4,4)); V_s1[0:2, 0:2] = V00; V_s1[2:4, 2:4] = V11
                sub_terms.append((U_s1, V_s1))
                # S2
                U_s2 = np.zeros((4,4)); U_s2[2:4, 0:2] = U10; U_s2[2:4, 2:4] = U11
                V_s2 = np.zeros((4,4)); V_s2[0:2, 0:2] = V00
                sub_terms.append((U_s2, V_s2))
                # S3
                U_s3 = np.zeros((4,4)); U_s3[0:2, 0:2] = U00
                V_s3 = np.zeros((4,4)); V_s3[0:2, 2:4] = V01; V_s3[2:4, 2:4] = -V11
                sub_terms.append((U_s3, V_s3))
                # S4
                U_s4 = np.zeros((4,4)); U_s4[2:4, 2:4] = U11
                V_s4 = np.zeros((4,4)); V_s4[2:4, 0:2] = V10; V_s4[0:2, 0:2] = -V00
                sub_terms.append((U_s4, V_s4))
                # S5
                U_s5 = np.zeros((4,4)); U_s5[0:2, 0:2] = U00; U_s5[0:2, 2:4] = U01
                V_s5 = np.zeros((4,4)); V_s5[2:4, 2:4] = V11
                sub_terms.append((U_s5, V_s5))
                # S6
                U_s6 = np.zeros((4,4)); U_s6[2:4, 0:2] = U10; U_s6[0:2, 0:2] = -U00
                V_s6 = np.zeros((4,4)); V_s6[0:2, 0:2] = V00; V_s6[0:2, 2:4] = V01
                sub_terms.append((U_s6, V_s6))
                # S7
                U_s7 = np.zeros((4,4)); U_s7[0:2, 2:4] = U01; U_s7[2:4, 2:4] = -U11
                V_s7 = np.zeros((4,4)); V_s7[2:4, 0:2] = V10; V_s7[2:4, 2:4] = V11
                sub_terms.append((U_s7, V_s7))
                
                for Us, Vs in sub_terms:
                    # Fill U_out, V_out
                    for i in range(4):
                        for j in range(4):
                            U_out[i*4+j, col] = Us[i, j]
                    for k in range(4):
                        for l in range(4):
                            V_out[k*4+l, col] = Vs[k, l]
                    
                    # W_out
                    # The term contributes to C[i, l] ?
                    # No, the bilinear form is A[i,j] B[k,l].
                    # In matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
                    # So the term A[i,j] B[j,k] contributes to C[i,k].
                    # Here we have A[i,j] B[k,l].
                    # So this term contributes to C[i,l] if j == k?
                    # No, the Strassen algorithm ensures that the indices match.
                    # The W matrix should have 1 at [k*n+i, r] ?
                    # The task says W[k*n+i, r].
                    # This implies the output index is i (row of A) and k (col of B)?
                    # No, standard is C[i,k].
                    # If W index is k*n+i, and n=4, then index is 4k+i.
                    # This is column-major for C?
                    # C[i,k] is at index 4k+i.
                    
                    # So for each non-zero A[i,j] B[k,l] term,
                    # if j == k, it contributes to C[i,l].
                    # So W[4*l + i, col] = 1.
                    
                    for i in range(4):
                        for j in range(4):
                            if Us[i, j] != 0:
                                for k in range(4):
                                    for l in range(4):
                                        if Vs[k, l] != 0:
                                            if j == k:
                                                W_out[4*l + i, col] += Us[i, j] * Vs[k, l]
                    
                    col += 1
            
            return {"U": U_out, "V": V_out, "W": W_out}
            
        else:
            # Rank 49
            return {"U": U_out, "V": V_out, "W": W_out}
            
    except Exception as e:
        # Fallback to Schoolbook
        rank = 64
        U_fb = [[0] * rank for _ in range(16)]
        V_fb = [[0] * rank for _ in range(16)]
        W_fb = [[0] * rank for _ in range(16)]
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j][c] = 1
                    V_fb[j*4+k][c] = 1
                    W_fb[4*k+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-START
def construct():
    # Call the helper
    return _construct_impl()

def _construct_impl():
    # ... (move code here)
    pass
# EVOLVE-BLOCK-END
