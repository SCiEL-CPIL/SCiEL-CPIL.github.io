import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    # Returns U, V, W for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: A: i*2+j, B: j*2+k, C: k*2+i
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1 # C21 += M2, C22 -= M2
    
    # M3 = A11(B21 - B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1 # C12 += M3, C21 += M3
    
    # M4 = A22(B12 - B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11 += M4, C21 += M4
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C11 -= M5, C12 += M5
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22 += M6
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11 += M7
    
    return U, V, W

def _compose_strassen(U1, V1, W1, U2, V2, W2):
    # Compose two Strassen decompositions
    # U1, V1, W1 for size N1
    # U2, V2, W2 for size N2
    # Result for size N1 * N2? No, this is for block structure.
    # This function is not quite right for direct composition.
    # I will generate 4x4x4 directly by applying 2x2 Strassen to blocks.
    pass

def _strassen_4x4():
    # Generate 4x4x4 decomposition using 2x2 Strassen on blocks
    # Rank 49
    U = np.zeros((16, 49))
    V = np.zeros((16, 49)) # B is 4x4, so 16 entries
    W = np.zeros((16, 49))
    
    # Get 2x2 Strassen
    U2, V2, W2 = _strassen_2x2()
    
    # The 4x4 matrices A, B, C are 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen formulas for blocks:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Each Mk is a 2x2 matrix product.
    # So Mk = Xk * Yk where Xk, Yk are 2x2.
    # We use the 2x2 Strassen decomposition for each Mk.
    
    # We need to map the block indices to flat indices.
    # A block (bi, bj) corresponds to flat indices:
    # Row: bi*2 + i, Col: bj*2 + j
    # Flat A index: (bi*2 + i)*4 + (bj*2 + j) = bi*8 + i*4 + bj*2 + j
    
    # Let's iterate over the 7 Strassen steps for the 2x2 block matrix.
    # For each step s (0..6), we have a multiplication of two 2x2 matrices.
    # Let the left matrix be L_s and right be R_s.
    # L_s and R_s are linear combinations of blocks of A.
    # R_s and ... wait, R_s is linear combination of blocks of B.
    
    # We need to construct the U, V, W vectors for the 49 rank-1 terms.
    # Each of the 7 block-multiplications has rank 7.
    # So we have 7 * 7 = 49 terms.
    
    # Let's define the block coefficients for A and B for each Strassen step.
    # Step 0 (M1):
    # A_blocks: A00 (1), A11 (1). Others 0.
    # B_blocks: B00 (1), B11 (1). Others 0.
    # Step 1 (M2):
    # A_blocks: A10 (1), A11 (1).
    # B_blocks: B00 (1).
    # ... and so on.
    
    # I will create a list of 7 tuples (A_coeffs, B_coeffs) where coeffs are 2x2 matrices of coefficients.
    # Then for each step, I will use the 2x2 Strassen decomposition to expand the product.
    
    a_coeffs = [
        np.array([[1, 0], [0, 1]]), # M1: A00+A11
        np.array([[0, 0], [1, 1]]), # M2: A10+A11
        np.array([[1, 0], [0, 0]]), # M3: A00
        np.array([[0, 0], [0, 1]]), # M4: A11
        np.array([[1, 1], [0, 0]]), # M5: A00+A01
        np.array([[-1, 0], [1, 0]]),# M6: A10-A00
        np.array([[0, 1], [0, -1]]) # M7: A01-A11
    ]
    
    b_coeffs = [
        np.array([[1, 0], [0, 1]]), # M1: B00+B11
        np.array([[1, 0], [0, 0]]), # M2: B00
        np.array([[0, -1], [1, 0]]),# M3: B10-B11
        np.array([[-1, 1], [0, 0]]),# M4: B01-B00
        np.array([[0, 0], [0, 1]]), # M5: B11
        np.array([[1, 1], [0, 0]]), # M6: B00+B01
        np.array([[0, 0], [1, 1]])  # M7: B10+B11
    ]
    
    # Now, for each step s, we compute L_s * R_s where L_s, R_s are 2x2 matrices.
    # The product is computed using Strassen 2x2.
    # The U, V, W for the 2x2 Strassen are U2, V2, W2.
    # We need to map the 2x2 indices to 4x4 flat indices.
    
    # Flat index for A element (i, j) in 4x4: i*4 + j.
    # Block (bi, bj) contains elements (bi*2+i, bj*2+j).
    # So A_flat[bi*2+i, bj*2+j] corresponds to A_block[bi, bj] element [i, j].
    
    # For a given step s, and a given 2x2 Strassen term t (0..6):
    # The term is U2[i, t] * A[i, j] * V2[j, t] * B[j, k] * W2[k, t] * C[k, i]
    # Wait, the Strassen decomposition is:
    # C = sum_t U_t A V_t B W_t^T? No.
    # The standard form is:
    # C_{ki} = sum_t W_{kt} (sum_j U_{jt} A_{ji} V_{kt} B_{jk})? No.
    # The tensor decomposition is:
    # T_{i,j,k} = sum_t U_{i,t} V_{j,t} W_{k,t}
    # For matrix multiplication C = A B:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{i,j,k} = A_{i,j} B_{j,k}.
    # Decomposition: A_{i,j} B_{j,k} = sum_t U_{i,t} V_{j,t} W_{k,t}
    # Here U corresponds to A indices, V to B indices, W to C indices.
    # My previous U, V, W definitions in _strassen_2x2 followed this.
    
    # So for step s:
    # A_eff[i, j] = sum_{bi, bj} a_coeffs[s][bi, bj] A_block[bi, bj][i, j]
    # B_eff[j, k] = sum_{bj, bk} b_coeffs[s][bj, bk] B_block[bj, bk][j, k]
    # C_eff[k, i] = sum_t U2[i, t] A_eff[i, j] V2[j, t] B_eff[j, k] W2[k, t]
    
    # We need to add these contributions to the global U, V, W.
    # Global U index: A_flat[i, j] = i*4 + j.
    # Global V index: B_flat[j, k] = j*4 + k.
    # Global W index: C_flat[k, i] = k*4 + i.
    
    col = 0
    for s in range(7):
        for t in range(7):
            # u_val = U2[i, t], v_val = V2[j, t], w_val = W2[k, t]
            # We need to expand the sums over blocks.
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # Contribution to U: U2[i, t] * a_coeffs[s][bi, bj] for all bi, bj
                        # Contribution to V: V2[j, t] * b_coeffs[s][bj, bk] for all bj, bk
                        # Contribution to W: W2[k, t]
                        
                        # Actually, the term is:
                        # sum_{bi, bj, bj2, bk} U2[i, t] a_coeffs[s][bi, bj] A_block[bi, bj][i, j] 
                        # * V2[j, t] b_coeffs[s][bj2, bk] B_block[bj2, bk][j, k] 
                        # * W2[k, t] C_block[bk, bi][k, i]
                        
                        # This is a rank-1 term in the 4x4x4 space?
                        # No, it's a sum of rank-1 terms.
                        # The rank-1 term is defined by specific flat indices.
                        
                        # Let's iterate over all combinations of blocks.
                        for bi in range(2):
                            for bj in range(2):
                                a_coeff = a_coeffs[s][bi, bj]
                                if a_coeff == 0: continue
                                for bj2 in range(2):
                                    for bk in range(2):
                                        b_coeff = b_coeffs[s][bj2, bk]
                                        if b_coeff == 0: continue
                                        
                                        # Flat indices
                                        a_flat = (bi*2 + i)*4 + (bj*2 + j)
                                        b_flat = (bj2*2 + j)*4 + (bk*2 + k)
                                        c_flat = (bk*2 + k)*4 + (bi*2 + i)
                                        
                                        u_val = U2[i, t] * a_coeff
                                        v_val = V2[j, t] * b_coeff
                                        w_val = W2[k, t]
                                        
                                        U[a_flat, col] = u_val
                                        V[b_flat, col] = v_val
                                        W[c_flat, col] = w_val
                                        
            col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    rank1 = 49 # Strassen 4x4x4
    rank2 = 49 # Strassen 4x4x4
    rank3 = 16 # Schoolbook 4x4x1
    total_rank = rank1 + rank2 + rank3
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1: B cols 0-3
    U1, V1, W1 = _strassen_4x4()
    # V1 is 4x4, flat j*4+k. We need j*9+k.
    # W1 is 4x4, flat k*4+i. We need k*4+i.
    for j in range(4):
        for k in range(4):
            old_idx = j*4 + k
            new_idx = j*9 + k
            V[new_idx, :rank1] = V1[old_idx, :]
    U[:, :rank1] = U1
    W[:, :rank1] = W1
    
    # Part 2: B cols 4-7
    U2, V2, W2 = _strassen_4x4()
    # V2 is 4x4, flat j*4+k. We need j*9+(k+4).
    # W2 is 4x4, flat k*4+i. We need (k+4)*4+i.
    for j in range(4):
        for k in range(4):
            old_idx = j*4 + k
            new_idx = j*9 + (k+4)
            V[new_idx, rank1:rank1+rank2] = V2[old_idx, :]
    U[:, rank1:rank1+rank2] = U2
    for k in range(4):
        for i in range(4):
            old_idx = k*4 + i
            new_idx = (k+4)*4 + i
            W[new_idx, rank1:rank1+rank2] = W2[old_idx, :]
            
    # Part 3: B col 8
    # Schoolbook 4x4x1
    # U3: 4x4x1 -> 16x16
    # V3: 4x1 -> 4x16? No, V is m*p x rank. m=4, p=1 part.
    # The V part for this sub-problem is just the B column.
    # V_flat for B col 8: j*9 + 8.
    # W_flat for C col 8: 8*4 + i.
    
    col_offset = rank1 + rank2
    col = col_offset
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j*9 + 8, col] = 1
            W[8*4 + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
