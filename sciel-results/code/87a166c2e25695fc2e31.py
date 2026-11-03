import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U[i*2 + j, r]
    V[j*2 + k, r]
    W[k*2 + i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11 contributions? 
    # Actually, W indices are (k, i). 
    # C00 gets M1, C11 gets M1.
    # C00 is (i=0, k=0). W index is k*2+i = 0*2+0 = 0.
    # C11 is (i=1, k=1). W index is k*2+i = 1*2+1 = 3.
    # So W[0,0]=1, W[3,0]=1.
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 gets M2. C10 is (i=1, k=0). W index 0*2+1 = 2.
    W[2, 1] = 1
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C01 gets M3. C01 is (i=0, k=1). W index 1*2+0 = 2.
    W[2, 2] = 1
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C00 gets M4, C10 gets M4.
    # C00 (i=0, k=0) -> W[0,3]=1
    # C10 (i=1, k=0) -> W[2,3]=1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 gets -M5, C01 gets M5.
    # C00 (i=0, k=0) -> W[0,4]=-1
    # C01 (i=0, k=1) -> W[2,4]=1
    W[0, 4] = -1
    W[2, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 gets M6.
    # C11 (i=1, k=1) -> W[3,5]=1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 gets M7.
    # C00 (i=0, k=0) -> W[0,6]=1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Checks that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i==j==k (diagonal of the product space? No, standard matmul check)
    
    Wait, the tensor is T[i, j, k] = 1 if the operation A[i,j] * B[j,k] contributes to C[i,k].
    The condition is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all valid (i,j,k) where the multiplication is defined.
    In standard matrix multiplication, every (i,j,k) triplet with i in 0..n-1, j in 0..m-1, k in 0..p-1 is valid.
    So we check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    R = U.shape[1]
    # Efficient check using broadcasting or einsum
    # T_est[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's construct the full tensor T_target (n, m, p) of ones.
    # And compute T_est.
    
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U_t = U.reshape(n, m, R)
    V_t = V.reshape(m, p, R)
    W_t = W.reshape(p, n, R)
    
    # T_est[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
    # This is tricky with standard einsum because indices are mixed.
    # Let's transpose W to (n, p, R) so it's W_t_trans[i, k, r]
    W_t_trans = W_t.transpose(1, 0, 2) # shape (n, p, R)
    
    # Now T_est[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t_trans[i, k, r]
    # We can sum over r.
    # Let's do it in steps to save memory if R is large, but R is small (<=128).
    
    # T_est = np.einsum('ijr,jkr,ikr->ijk', U_t, V_t, W_t_trans)
    # This might be slow for large dimensions, but 4x4x8 is tiny.
    
    T_est = np.einsum('ijr,jkr,ikr->ijk', U_t, V_t, W_t_trans)
    
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_est, T_target, atol=1e-9)

def _strassen_4x4_4():
    """
    Construct rank-49 decomposition for 4x4x4 using Strassen recursion.
    Returns U, V, W where U is (16, 49), V is (16, 49), W is (16, 49).
    """
    U2, V2, W2 = _strassen_2x2_2()
    R2 = 7
    R4 = R2 * R2 # 49
    
    U4 = np.zeros((16, R4))
    V4 = np.zeros((16, R4))
    W4 = np.zeros((16, R4))
    
    # Block structure:
    # A is 4x4, viewed as 2x2 blocks of 2x2.
    # A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), etc.
    # U index: i*4 + j.
    # i = 2*I + i_loc, j = 2*J + j_loc.
    
    # B is 4x4, viewed as 2x2 blocks of 2x2.
    # B00 (rows 0-1, cols 0-1), etc.
    # V index: j*4 + k.
    # j = 2*J + j_loc, k = 2*K + k_loc.
    
    # C is 4x4, viewed as 2x2 blocks of 2x2.
    # C index: k*4 + i.
    # k = 2*K + k_loc, i = 2*I + i_loc.
    
    # Strassen block products Mq (q=0..6) involve specific combinations of A blocks and B blocks.
    # Mq is a 2x2 matrix product.
    # Mq = A_q * B_q where A_q and B_q are 2x2 linear combinations of the 2x2 blocks.
    
    # Let's define the coefficients for A and B blocks for each Strassen product.
    # A blocks: A00, A01, A10, A11 (indices (I,J))
    # B blocks: B00, B01, B10, B11 (indices (J,K))
    
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # Coefficients for A blocks (I, J) for each Mq:
    # q=0: A00(1), A11(1)
    # q=1: A10(1), A11(1)
    # q=2: A00(1)
    # q=3: A11(1)
    # q=4: A00(1), A01(1)
    # q=5: A10(1), A00(-1)
    # q=6: A01(1), A11(-1)
    
    # Coefficients for B blocks (J, K) for each Mq:
    # q=0: B00(1), B11(1)
    # q=1: B00(1)
    # q=2: B01(1), B11(-1)
    # q=3: B10(1), B00(-1)
    # q=4: B11(1)
    # q=5: B00(1), B01(1)
    # q=6: B10(1), B11(1)
    
    # C blocks combination:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each Mq, we decompose the 2x2 matrix product into 7 scalar products.
    # The scalar products are indexed by r_loc (0..6).
    # For a given Mq, the scalar product r_loc uses:
    #   U2[i_loc, r_loc] for A block element (I, J)
    #   V2[j_loc, r_loc] for B block element (J, K)
    #   W2[k_loc, r_loc] for C block element (K, I)
    
    # We need to sum contributions from all Mq to the C blocks.
    
    # Let's pre-compute the A and B block coefficients.
    A_coeffs = [
        {(0,0): 1, (1,1): 1}, # q=0
        {(1,0): 1, (1,1): 1}, # q=1
        {(0,0): 1},           # q=2
        {(1,1): 1},           # q=3
        {(0,0): 1, (0,1): 1}, # q=4
        {(1,0): 1, (0,0): -1},# q=5
        {(0,1): 1, (1,1): -1} # q=6
    ]
    
    B_coeffs = [
        {(0,0): 1, (1,1): 1}, # q=0
        {(0,0): 1},           # q=1
        {(0,1): 1, (1,1): -1},# q=2
        {(1,0): 1, (0,0): -1},# q=3
        {(1,1): 1},           # q=4
        {(0,0): 1, (0,1): 1}, # q=5
        {(1,0): 1, (1,1): 1}  # q=6
    ]
    
    C_coeffs = [
        # M0, M1, M2, M3, M4, M5, M6
        # C00
        {(0,0): [1, 0, 0, 1, -1, 0, 1]},
        # C01
        {(0,1): [0, 0, 1, 0, 1, 0, 0]},
        # C10
        {(1,0): [0, 1, 0, 1, 0, 0, 0]},
        # C11
        {(1,1): [1, -1, 1, 0, 0, 1, 0]}
    ]
    
    # Iterate over q (Strassen product) and r_loc (scalar product within Mq)
    for q in range(7):
        for r_loc in range(7):
            r4 = q * 7 + r_loc
            
            # Get A and B block coefficients for this q
            for (I, J), a_coeff in A_coeffs[q].items():
                for (Jb, K), b_coeff in B_coeffs[q].items():
                    if J != Jb:
                        continue # A and B must share the middle index J
                    
                    # For each local element in A block (I, J) and B block (J, K)
                    # U2[i_loc, r_loc] corresponds to A element (I, J) at local position (i_loc, j_loc)?
                    # No, U2 is for 2x2x2.
                    # U2 indices: i*2 + j. Here i, j are local indices within the 2x2 block.
                    # So for a specific (I, J) block, the element (i_loc, j_loc) has U4 index:
                    # i_global = 2*I + i_loc
                    # j_global = 2*J + j_loc
                    # U4_index = i_global * 4 + j_global
                    
                    # Similarly for V:
                    # j_global = 2*J + j_loc
                    # k_global = 2*K + k_loc
                    # V4_index = j_global * 4 + k_global
                    
                    # And for W:
                    # k_global = 2*K + k_loc
                    # i_global = 2*I + i_loc
                    # W4_index = k_global * 4 + i_global
                    
                    # We need to add a_coeff * U2[i_loc, r_loc] to U4[U4_index, r4]
                    # But U2[i_loc, r_loc] is non-zero only for specific i_loc, j_loc?
                    # U2 is defined for all i_loc, j_loc.
                    
                    # Let's iterate over local indices i_loc, j_loc, k_loc
                    for i_loc in range(2):
                        for j_loc in range(2):
                            # Check if U2[i_loc*2+j_loc, r_loc] is non-zero
                            u_val = U2[i_loc*2 + j_loc, r_loc]
                            if u_val != 0:
                                i_global = 2*I + i_loc
                                j_global = 2*J + j_loc
                                U4[i_global*4 + j_global, r4] += a_coeff * u_val
                                
                    for j_loc in range(2):
                        for k_loc in range(2):
                            v_val = V2[j_loc*2 + k_loc, r_loc]
                            if v_val != 0:
                                j_global = 2*J + j_loc
                                k_global = 2*K + k_loc
                                V4[j_global*4 + k_global, r4] += b_coeff * v_val
                                
                    # Now for W, we need to add to the correct C blocks.
                    # The C block (K, I) receives contribution from Mq with coefficient c_coeff[K, I, q]
                    for k_loc in range(2):
                        for i_loc in range(2):
                            w_val = W2[k_loc*2 + i_loc, r_loc]
                            if w_val != 0:
                                k_global = 2*K + k_loc
                                i_global = 2*I + i_loc
                                # Find the coefficient for C block (K, I) from Mq
                                # C_coeffs is a list of dicts mapping (K, I) to list of coeffs for q=0..6
                                # We need to sum over C blocks?
                                # No, for a fixed (I, J, K) from A and B, the C block is (K, I).
                                # So we look up C_coeffs for (K, I) and get the coefficient for q.
                                # But C_coeffs is defined as {(K, I): [coeffs for q=0..6]}
                                # So we need to find the entry for (K, I).
                                
                                # Wait, my C_coeffs structure is:
                                # C_coeffs is a list of 4 dicts? No, I defined it as a list of dicts mapping (K,I) to coeffs.
                                # Actually, I defined C_coeffs as a list of dicts, but each dict has one key.
                                # Let's flatten it: C_coeff_map[(K, I)][q]
                                
                                c_coeff = 0
                                for d in C_coeffs:
                                    if (K, I) in d:
                                        c_coeff = d[(K, I)][q]
                                        break
                                
                                if c_coeff != 0:
                                    W4[k_global*4 + i_global, r4] += c_coeff * w_val

    return U4, V4, W4

def _find_rank20_444():
    """
    Attempt to find a rank-20 decomposition for 4x4x4.
    This is a known open problem or hard problem. 
    If we find one, we can use it to build 4x4x8 with rank 40.
    Since it's hard, we might just return the Strassen rank-49 decomposition.
    """
    # For now, fall back to Strassen
    return _strassen_4x4_4()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # We are multiplying 4x4 by 4x8.
    # This can be viewed as two independent 4x4x4 multiplications?
    # A is 4x4. B is 4x8, which is two 4x4 blocks: B_left (cols 0-3) and B_right (cols 4-7).
    # C is 4x8, which is two 4x4 blocks: C_left and C_right.
    # C_left = A * B_left
    # C_right = A * B_right
    
    # So we can use the 4x4x4 decomposition twice.
    # Rank will be R4 + R4 = 2 * R4.
    
    U4, V4, W4 = _strassen_4x4_4()
    R4 = U4.shape[1]
    
    R_total = 2 * R4
    
    U = np.zeros((n*m, R_total))
    V = np.zeros((m*p, R_total))
    W = np.zeros((p*n, R_total))
    
    # First block: C_left
    U[:, 0:R4] = U4
    V[:, 0:R4] = V4 # V4 is for B_left (j, k where k is 0..3)
    # W for C_left: C_left[k, i] corresponds to W[k*n+i, r]
    # W4 is (p*n, R4) where p=4, n=4. W4 index is k*4+i.
    # This matches W index k*n+i.
    W[:, 0:R4] = W4
    
    # Second block: C_right
    # B_right is cols 4-7.
    # V index is j*p + k. For B_right, k is 4..7.
    # V4 is for k=0..3. We need to shift the k index by 4.
    # V_right[j, k'] = V4[j, k'] where k' = k - 4.
    # V index in full V: j*p + k = j*4 + (k'+4) = 4j + 4 + k'.
    # V4 index: j*4 + k'.
    # So V_right[4j + 4 + k'] = V4[j*4 + k'].
    
    # Let's construct V for the right block.
    V_right_block = np.zeros((m*p, R4))
    for j in range(m):
        for k_prime in range(4):
            V_right_block[j*4 + (k_prime+4), :] = V4[j*4 + k_prime, :]
            
    V[:, R4:2*R4] = V_right_block
    
    # W for C_right: C_right[k, i].
    # W index is k*n + i.
    # W4 is the same structure, just for the second C block.
    # The decomposition is identical, so W_right = W4.
    W[:, R4:2*R4] = W4
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_school = np.zeros((n*m, schoolbook_rank))
        V_school = np.zeros((m*p, schoolbook_rank))
        W_school = np.zeros((p*n, schoolbook_rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
                    
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
