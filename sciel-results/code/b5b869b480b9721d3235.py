"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reconstruct the tensor
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape for batch matmul
    U_r = U.reshape(n, m, -1)  # (n, m, rank)
    V_r = V.reshape(m, p, -1)  # (m, p, rank)
    W_r = W.reshape(p, n, -1)  # (p, n, rank)
    
    # For each (i, j, k), we need sum over c
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We can compute this as: for each c, the outer product of vectors
    # But more efficiently: 
    # Let's iterate over c and add contributions
    # Actually, let's use einsum
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a bit tricky with einsum because of the different indices
    
    # Alternative: 
    # T = sum_c U[:, :, c] * V[:, :, c] * W[:, :, c]  (element-wise in the right places)
    # Let's just use a loop over rank if it's small enough, or a smarter approach
    
    # Actually, the most straightforward way:
    # Create a 4D tensor: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We can do this by:
    # 1. For each c, compute the contribution
    # 2. Sum
    
    # Let's use a more efficient approach with broadcasting
    # T[i,j,k] = sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c]
    # Let M[i,j,k] = sum_c U[i,j,c] * V[j,k,c]
    # Then T[i,j,k] = sum_k' M[i,j,k'] * W[k',i,k']  -- no, this is wrong
    
    # Let's just do it directly
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U_r[:, :, c]  # (n, m)
        V_c = V_r[:, :, c]  # (m, p)
        W_c = W_r[:, :, c]  # (p, n)
        # For each (i, j, k):
        # Contribution = U_c[i,j] * V_c[j,k] * W_c[k,i]
        # We can compute this as:
        # Temp[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
        # This is: U_c[i,j] * (V_c[j,k] * W_c[k,i])
        # Let's compute V_c @ W_c first? No, indices don't match for matmul directly
        # V_c is (m,p), W_c is (p,n)
        # V_c @ W_c would be (m,n), but we need (m,p) for V_c[j,k] and (p,n) for W_c[k,i]
        # 
        # Let's use einsum for the triple product:
        # np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
        T += np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
    
    # Now T should equal the identity tensor for matrix multiplication
    # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] should be 1 if this is the "identity" in the sense that
    # when we contract with A and B, we get C
    # Actually, the decomposition should satisfy:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta_{correct}
    # For matrix multiplication, the tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition gives:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k? No.
    
    # Let me re-read the task. The tensor for matrix multiplication M_{n,m,p} is:
    # T_{i,j,k} = 1 for all i,j,k (in the standard basis)
    # Wait, no. The matrix multiplication tensor is:
    # T(A,B) = A*B
    # In tensor form: T_{i,j,k} where i indexes rows of A, j columns of A/rows of B, k columns of B
    # T_{i,j,k} = 1 (it's the identity tensor in the sense that it maps (A,B) to A*B)
    # Actually, T_{i,j,k} = delta_{i,j,k} doesn't make sense for 3 indices.
    
    # The correct statement: the tensor T in R^{n x m x p} is defined by:
    # T_{i,j,k} = 1 if we're multiplying A[i,j] * B[j,k] to get C[i,k]
    # So T_{i,j,k} = 1 for all valid i,j,k.
    
    # So we need: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol):
        diff = np.max(np.abs(T - expected))
        raise ValueError(f"Verification failed: max diff = {diff}")
    
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication.
    
    We use the decomposition:
    T_{3,4,4} = T_{2,2,2} ⊗ T_{2,2,2} ⊕ T_{1,2,2} ⊗ T_{2,2,2}
    
    Where:
    - T_{2,2,2} has rank 7 (Strassen)
    - T_{1,2,2} has rank 4 (schoolbook)
    
    Total rank: 7*4 + 4*4 = 28 + 16 = 44
    
    The Kronecker product of decompositions:
    If T1 has rank r1 with (U1,V1,W1) and T2 has rank r2 with (U2,V2,W2),
    then T1⊗T2 has rank r1*r2 with:
    U = U1 ⊗ U2, V = V1 ⊗ V2, W = W2 ⊗ W1 (note the order for W)
    
    Wait, let me be careful. The tensor product T⊗S for matrix multiplication:
    (T⊗S)(A⊗B, C⊗D) = (AC)⊗(BD)
    
    The Kronecker product of the factor matrices:
    If T = sum_{c=1}^{r_T} u_c ⊗ v_c ⊗ w_c
    and S = sum_{d=1}^{r_S} a_d ⊗ b_d ⊗ e_d
    
    Then T⊗S = sum_{c,d} (u_c⊗a_d) ⊗ (v_c⊗b_d) ⊗ (w_c⊗e_d)
    
    So U_{cd} = u_c ⊗ a_d, V_{cd} = v_c ⊗ b_d, W_{cd} = w_c ⊗ e_d
    
    The dimensions:
    T_{n1,m1,p1} ⊗ T_{n2,m2,p2} = T_{n1*n2, m1*m2, p1*p2}
    
    So for T_{3,4,4}:
    T_{3,4,4} = (T_{2,2,2} ⊕ T_{1,2,2}) ⊗ T_{2,2,2}
    
    Where T_{2,2,2} ⊕ T_{1,2,2} is a direct sum of T_{2,2,2} and T_{1,2,2}
    
    The direct sum T_A ⊕ T_B for T_A: n1 x m1 x p1, T_B: n2 x m2 x p2:
    T = T_A ⊕ T_B: n x m x p where n=n1+n2, m=m1+m2, p=p1+p2? No.
    
    Actually, for matrix multiplication, we can decompose:
    T_{3,4,4} as follows:
    
    A is 3x4, B is 4x4, C = A*B is 3x4.
    
    Split A into A_top (2x4) and A_bottom (1x4)
    Split B into B_left (4x2) and B_right (4x2)
    
    Then C_top_left = A_top * B_left (2x2)
    C_top_right = A_top * B_right (2x2)
    C_bottom_left = A_bottom * B_left (1x2)
    C_bottom_right = A_bottom * B_right (1x2)
    
    So T_{3,4,4} = T_{2,2,2} (for top-left) + T_{2,2,2} (for top-right) + T_{1,2,2} (for bottom-left) + T_{1,2,2} (for bottom-right)
    
    But this is not quite right because the decompositions need to be on the same "blocks".
    
    Let me think again. The standard way to decompose using Kronecker products:
    
    T_{3,4,4} can be seen as:
    T_{3,4,4} = T_{2,2,2} ⊗ T_{1,2,2} ⊕ T_{1,2,2} ⊗ T_{2,2,2}
    
    Where:
    - First term: top 2 rows of A, first 2 cols of B -> 2x2 result, combined with 1x2 for the rest?
    
    Actually, let me use a cleaner decomposition:
    
    T_{3,4,4} = T_{2,2,2} ⊗ T_{2,2,2} ⊕ T_{1,2,2} ⊗ T_{2,2,2}
    
    This gives:
    - First part: 2x2 x 2x2 matrix mult, rank 7*4=28, contributes to top 2 rows of C
    - Second part: 1x2 x 2x2 matrix mult, rank 4*4=16, contributes to bottom row of C
    
    Total: 3 rows of C, 4x4 inputs, rank 44.
    
    But wait, T_{2,2,2} ⊗ T_{2,2,2} = T_{4,4,4}, not T_{2,4,2}.
    
    Let me reconsider. I think the correct decomposition is:
    
    T_{3,4,4} = T_{3,2,2} ⊗ T_{1,2,2}
    
    Where T_{3,2,2} has some rank r, and T_{1,2,2} has rank 4.
    
    T_{3,2,2}: 3x2 matrix times 2x2 matrix = 3x2 matrix.
    
    T_{3,2,2} = T_{2,2,2} ⊕ T_{1,2,2}
    Rank = 7 + 4 = 11.
    
    So T_{3,4,4} = T_{3,2,2} ⊗ T_{1,2,2}, rank = 11 * 4 = 44.
    
    This makes sense! The 3x4 x 4x4 multiplication can be seen as:
    - First multiply each 2-column block of B with A: gives T_{3,2,2} for each pair of columns
    - Then combine the results: T_{1,2,2} combines the two 3x2 blocks into 3x4
    
    Actually, I think I have it backwards. Let me just implement it carefully.
    
    For T_{n1,m1,p1} ⊗ T_{n2,m2,p2} = T_{n1*n2, m1*m2, p1*p2}:
    
    U_total: (n1*n2)*(m1*m2) x (r1*r2)
    V_total: (m1*m2)*(p1*p2) x (r1*r2)  
    W_total: (p1*p2)*(n1*n2) x (r1*r2)
    
    For the Kronecker product:
    U_total[i1*m2+n1? , ...] -- I need to get the indexing right.
    
    Let me just code it up and verify.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Define Strassen for 2x2x2 (rank 7)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U: 2x2 x 7, indexed by (i,j) where i=row of A, j=col of A
    # V: 2x2 x 7, indexed by (j,k) where j=row of B, k=col of B  
    # W: 2x2 x 7, indexed by (k,i) where k=col of B, i=row of A (output)
    
    U_222 = np.zeros((4, 7))  # 2*2 rows
    V_222 = np.zeros((4, 7))  # 2*2 rows
    W_222 = np.zeros((4, 7))  # 2*2 rows
    
    # M1: (A11+A22)(B11+B22)
    # U: A11=1, A22=1 -> U[0,0]=1, U[3,0]=1 (row 0 col 0, row 1 col 1 in 2x2)
    # V: B11=1, B22=1 -> V[0,0]=1, V[3,0]=1
    # W: C11 gets +1, C22 gets +1 -> W[0,0]=1, W[3,0]=1 (k=0,i=0 and k=1,i=1)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # M2: (A21+A22)B11
    U_222[2, 1] = 1; U_222[3, 1] = 1  # A21=row1,col0; A22=row1,col1
    V_222[0, 1] = 1  # B11
    W_222[2, 1] = 1  # C21: k=1,i=1 -> index k*n+i = 1*2+1 = 3? Wait.
    # W is indexed by (k,i) where k is col of B (0 or 1), i is row of A (0 or 1)
    # Output C[i,k], so W[k,i]
    # C21 means i=1, k=1, so W[1,1] = W[1*2+1] = W[3]
    # Let me redo the indexing.
    # U: row i (0,1), col j (0,1) -> index i*2+j
    # V: row j (0,1), col k (0,1) -> index j*2+k
    # W: row k (0,1), col i (0,1) -> index k*2+i
    
    # Let me restart with clearer indexing.
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    def u_idx(i, j): return i * 2 + j
    def v_idx(j, k): return j * 2 + k
    def w_idx(k, i): return k * 2 + i
    
    # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    U_222[u_idx(0,0), 0] = 1; U_222[u_idx(1,1), 0] = 1
    V_222[v_idx(0,0), 0] = 1; V_222[v_idx(1,1), 0] = 1
    W_222[w_idx(0,0), 0] = 1; W_222[w_idx(1,1), 0] = 1
    
    # M2: (A21+A22)B11 -> C21 += 1
    U_222[u_idx(1,0), 1] = 1; U_222[u_idx(1,1), 1] = 1
    V_222[v_idx(0,0), 1] = 1
    W_222[w_idx(1,1), 1] = 1
    
    # M3: A11(B12-B22) -> C12 += 1
    U_222[u_idx(0,0), 2] = 1
    V_222[v_idx(0,1), 2] = 1; V_222[v_idx(1,1), 2] = -1
    W_222[w_idx(1,0), 2] = 1
    
    # M4: A22(B21-B11) -> C11 += 1, C21 += 1
    U_222[u_idx(1,1), 3] = 1
    V_222[v_idx(1,0), 3] = 1; V_222[v_idx(0,0), 3] = -1
    W_222[w_idx(0,0), 3] = 1; W_222[w_idx(1,1), 3] = 1
    
    # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
    U_222[u_idx(0,0), 4] = 1; U_222[u_idx(0,1), 4] = 1
    V_222[v_idx(1,1), 4] = 1
    W_222[w_idx(0,0), 4] = -1; W_222[w_idx(1,0), 4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C22 += 1
    U_222[u_idx(1,0), 5] = 1; U_222[u_idx(0,0), 5] = -1
    V_222[v_idx(0,0), 5] = 1; V_222[v_idx(0,1), 5] = 1
    W_222[w_idx(1,1), 5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C11 += 1, C22 -= 1
    U_222[u_idx(0,1), 6] = 1; U_222[u_idx(1,1), 6] = -1
    V_222[v_idx(1,0), 6] = 1; V_222[v_idx(1,1), 6] = 1
    W_222[w_idx(0,0), 6] = 1; W_222[w_idx(1,1), 6] = -1
    
    # Verify 2x2x2
    _verify(U_222.tolist(), V_222.tolist(), W_222.tolist(), 2, 2, 2)
    
    # Schoolbook for 1x2x2 (rank 4)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    U_122 = np.zeros((2, 4))  # 1*2 rows
    V_122 = np.zeros((4, 4))  # 2*2 rows
    W_122 = np.zeros((2, 4))  # 2*1 rows
    
    def u1_idx(i, j): return i * 2 + j  # i=0 always
    def v1_idx(j, k): return j * 2 + k
    def w1_idx(k, i): return k * 1 + i  # i=0 always, so just k
    
    # C11 = A11*B11 + A12*B21
    U_122[u1_idx(0,0), 0] = 1
    V_122[v1_idx(0,0), 0] = 1
    W_122[w1_idx(0,0), 0] = 1
    
    U_122[u1_idx(0,1), 1] = 1
    V_122[v1_idx(1,0), 1] = 1
    W_122[w1_idx(0,0), 1] = 1
    
    # C12 = A11*B12 + A12*B22
    U_122[u1_idx(0,0), 2] = 1
    V_122[v1_idx(0,1), 2] = 1
    W_122[w1_idx(1,0), 2] = 1
    
    U_122[u1_idx(0,1), 3] = 1
    V_122[v1_idx(1,1), 3] = 1
    W_122[w1_idx(1,0), 3] = 1
    
    _verify(U_122.tolist(), V_122.tolist(), W_122.tolist(), 1, 2, 2)
    
    # Now combine: T_{3,4,4} = T_{3,2,2} ⊗ T_{1,2,2}
    # Where T_{3,2,2} = T_{2,2,2} ⊕ T_{1,2,2}
    
    # First, form T_{3,2,2} by direct sum of T_{2,2,2} and T_{1,2,2}
    # T_{3,2,2}: n=3, m=2, p=2, rank = 7+4 = 11
    
    # For direct sum T_A ⊕ T_B:
    # U_total: (nA*mA + nB*mB) x (rA + rB)
    # First rA columns from U_A, next rB columns from U_B
    
    n322, m322, p322 = 3, 2, 2
    r322 = 11
    
    U_322 = np.zeros((n322 * m322, r322))
    V_322 = np.zeros((m322 * p322, r322))
    W_322 = np.zeros((p322 * n322, r322))
    
    # Copy T_{2,2,2} into first part
    # T_{2,2,2} has n=2, m=2, p=2
    U_322[:4, :7] = U_222
    V_322[:4, :7] = V_222
    W_322[:4, :7] = W_222
    
    # Copy T_{1,2,2} into second part, offset by 2 in n-index
    # T_{1,2,2} has n=1, m=2, p=2
    # Its U has 1*2=2 rows, V has 2*2=4 rows, W has 2*1=2 rows
    # Offset in U: row 2 (since first block had 2 rows)
    U_322[4:6, 7:] = U_122
    V_322[4:8, 7:] = V_122  # Wait, V_122 has 4 rows, but we need to place them correctly
    # Actually, for the direct sum, the indices should be:
    # U: first nA*mA rows for A, next nB*mB rows for B
    # V: first mA*pA rows for A, next mB*pB rows for B
    # W: first pA*nA rows for A, next pB*nB rows for B
    
    # T_{2,2,2}: nA=2, mA=2, pA=2 -> U:4 rows, V:4 rows, W:4 rows
    # T_{1,2,2}: nB=1, mB=2, pB=2 -> U:2 rows, V:4 rows, W:2 rows
    
    # So:
    # U_322: rows 0-3 from T_222, rows 4-5 from T_122
    # V_322: rows 0-3 from T_222, rows 4-7 from T_122
    # W_322: rows 0-3 from T_222, rows 4-5 from T_122
    
    V_322[4:8, 7:] = V_122
    W_322[4:6, 7:] = W_122
    
    _verify(U_322.tolist(), V_322.tolist(), W_322.tolist(), 3, 2, 2)
    
    # Now Kronecker product with T_{1,2,2}
    # T_{3,4,4} = T_{3,2,2} ⊗ T_{1,2,2}
    # n = 3*1 = 3, m = 2*2 = 4, p = 2*2 = 4
    # rank = 11 * 4 = 44
    
    n_final, m_final, p_final = 3, 4, 4
    r_final = 44
    
    # For Kronecker product T_A ⊗ T_B:
    # U_total: (nA*nB)*(mA*mB) x (rA*rB)
    # V_total: (mA*mB)*(pA*pB) x (rA*rB)
    # W_total: (pA*pB)*(nA*nB) x (rA*rB)
    # 
    # Indexing:
    # U_total[(iA*nB+iB)*(mA*mB)+(jA*mB+jB), cA*rB+cB] = U_A[iA*mA+jA, cA] * U_B[iB*mB+jB, cB]
    # V_total[(jA*mB+jB)*(pA*pB)+(kA*pB+kB), cA*rB+cB] = V_A[jA*pA+kA, cA] * V_B[jB*pB+kB, cB]
    # W_total[(kA*pB+kB)*(nA*nB)+(iA*nB+iB), cA*rB+cB] = W_A[kA*nA+iA, cA] * W_B[kB*nB+iB, cB]
    
    U_final = np.zeros((n_final * m_final, r_final))
    V_final = np.zeros((m_final * p_final, r_final))
    W_final = np.zeros((p_final * n_final, r_final))
    
    nA, mA, pA, rA = 3, 2, 2, 11
    nB, mB, pB, rB = 1, 2, 2, 4
    
    for cA in range(rA):
        for cB in range(rB):
            c = cA * rB + cB
            
            # U
            for iA in range(nA):
                for jA in range(mA):
                    uA = U_322[iA * mA + jA, cA]
                    if uA == 0:
                        continue
                    for iB in range(nB):
                        for jB in range(mB):
                            uB = U_122[iB * mB + jB, cB]
                            if uB == 0:
                                continue
                            i = iA * nB + iB
                            j = jA * mB + jB
                            U_final[i * m_final + j, c] = uA * uB
            
            # V
            for jA in range(mA):
                for kA in range(pA):
                    vA = V_322[jA * pA + kA, cA]
                    if vA == 0:
                        continue
                    for jB in range(mB):
                        for kB in range(pB):
                            vB = V_122[jB * pB + kB, cB]
                            if vB == 0:
                                continue
                            j = jA * mB + jB
                            k = kA * pB + kB
                            V_final[j * p_final + k, c] = vA * vB
            
            # W
            for kA in range(pA):
                for iA in range(nA):
                    wA = W_322[kA * nA + iA, cA]
                    if wA == 0:
                        continue
                    for kB in range(pB):
                        for iB in range(nB):
                            wB = W_122[kB * nB + iB, cB]
                            if wB == 0:
                                continue
                            k = kA * pB + kB
                            i = iA * nB + iB
                            W_final[k * n_final + i, c] = wA * wB
    
    # Verify the final decomposition
    _verify(U_final.tolist(), V_final.tolist(), W_final.tolist(), 3, 4, 4)
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
