import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices: 
    U[i*m+j, r] where m=2. 
    V[j*p+k, r] where p=2.
    W[k*n+i, r] where n=2.
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor decomposition form:
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For 2x2x2:
    # i, j, k in {0, 1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    # Let's define the 7 terms explicitly.
    # Each term r corresponds to one product.
    # U_r, V_r, W_r are vectors of length 4 (for 2x2 matrices flattened).
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    # A00 is A[0,0] -> U[0,0]=1. A11 is A[1,1] -> U[3,0]=1.
    # B00 is B[0,0] -> V[0,0]=1. B11 is B[1,1] -> V[3,0]=1.
    # C00 gets +1, C11 gets +1.
    # W[k*n+i, r]. 
    # For C00: i=0, j=0. k sums over 0,1. 
    # The product is M1. 
    # C00 = sum_k ... 
    # Actually, the decomposition defines the tensor T.
    # T[i,j,k] = U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    
    # Let's fill U, V, W for each of the 7 Strassen steps.
    
    # r=0: M1
    # A part: A00 + A11. U[0,0]=1, U[3,0]=1.
    # B part: B00 + B11. V[0,0]=1, V[3,0]=1.
    # C part: C00 + C11. 
    # C00 corresponds to i=0, j=0. W index for i=0, j=0 is W[k*2+0, 0].
    # C11 corresponds to i=1, j=1. W index for i=1, j=1 is W[k*2+1, 0].
    # So W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1? 
    # No, W is indexed by k*n+i. 
    # For C[i,j], we sum over k.
    # C[0,0] = sum_k T[0,0,k] A[0,k] B[k,0].
    # In Strassen, C00 = M1 + M4 - M5 + M7.
    # M1 contributes to C00.
    # The term M1 is (A00+A11)(B00+B11).
    # This product is a scalar. It adds to C00 and C11.
    # So for r=0:
    # U[0,0]=1, U[3,0]=1.
    # V[0,0]=1, V[3,0]=1.
    # W[0,0]=1 (k=0,i=0 -> C00), W[2,0]=1 (k=1,i=0 -> C00? No, k is summed in T? 
    # Wait. The tensor T[i,j,k] is the coefficient of A[i,k]B[k,j] in C[i,j].
    # In Strassen, we don't just multiply A[i,k]B[k,j]. We form linear combinations.
    # The decomposition T = sum U_r V_r W_r means:
    # sum_r U_r[i*m+j] V_r[j*p+k] W_r[k*n+i] = T[i,j,k]
    # For 2x2, T[i,j,k] is the matmul tensor.
    # T[0,0,0]=1, T[0,0,1]=0? No.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # So T[0,0,0]=1, T[0,0,1]=1? No, T[0,0,k] is coeff of A[0,k]B[k,0].
    # So T[0,0,0]=1, T[0,0,1]=1.
    # T[0,1,0]=1, T[0,1,1]=1.
    # T[1,0,0]=1, T[1,0,1]=1.
    # T[1,1,0]=1, T[1,1,1]=1.
    # Wait, T[i,j,k] = 1 for all i,j,k in 2x2?
    # If T is all ones, rank is 1? No, tensor rank of all-ones 2x2x2 is 1?
    # No, the matmul tensor is NOT all ones.
    # C[i,j] = sum_k A[i,k] B[k,j].
    # The tensor M is defined by M[i,j,k] = 1 if the term A[i,k]B[k,j] appears.
    # Yes, for 2x2, every term appears.
    # So M[i,j,k] = 1 for all i,j,k.
    # But the rank of the all-ones tensor 2x2x2 is 1?
    # Let's check. 1*1*1 = 1. Yes.
    # But Strassen says rank 7.
    # The confusion is that the "tensor" in tensor rank decomposition is not just the indicator of terms.
    # It is the tensor such that <M, A, B> = trace(C^T A B).
    # M[i,j,k] = 1 for all i,j,k.
    # Why is rank 7?
    # Because we are decomposing the *bilinear* map? No, trilinear.
    # Ah, the matrix multiplication tensor $M_{2,2,2}$ has entries $M_{i,j,k} = \delta_{i+k, j}$? No.
    # Let's stick to the standard definition used in the seed code.
    # Seed code: U[i*m+j], V[j*p+k], W[k*n+i].
    # And it checks: sum_r U V W == 1 for all i,j,k.
    # This implies the target tensor is ALL ONES.
    # If the target tensor is all ones, the rank is 1 (U=1, V=1, W=1).
    # But the seed code returns rank 8 (schoolbook).
    # And the task says "score is the rank".
    # If rank 1 was possible, the score would be 1/125.
    # But the seed score is 1.0, implying rank 125 is the baseline.
    # This implies my understanding of "target tensor" is wrong.
    
    # Let's re-read the seed code verification.
    # "C[i,j] = sum_r W[k*n+i, r] * (sum_k U[i*m+k, r] A[i,k]) * (sum_k V[k*p+j, r] B[k,j])"
    # This formula is weird. It sums k inside U and V, but W depends on k?
    # No, the seed code verification is:
    # f_r = sum_k U[i*m+k, r] A[i,k]
    # g_r = sum_k V[k*p+j, r] B[k,j]
    # C[i,j] += W[i*n+j, r] * f_r * g_r  (Note: W index is i*n+j, not k*n+i)
    
    # If this is the definition, then:
    # C[i,j] = sum_r W[i*n+j, r] (sum_k U[i*m+k, r] A[i,k]) (sum_l V[l*p+j, r] B[l,j])
    # This is a standard tensor decomposition of the matmul bilinear form.
    # The "tensor" being decomposed is not the trilinear form directly, but the bilinear map A,B -> C.
    # The rank is the number of scalar multiplications.
    
    # For 2x2, Strassen rank 7 is correct for this definition.
    # My previous "all ones" tensor was the trilinear form, which is different.
    
    # So, for 2x2 Strassen:
    # We need U (4x7), V (4x7), W (4x7) such that:
    # For all i,j in {0,1}:
    # C[i,j] = sum_r W[i*2+j, r] * (sum_k U[i*2+k, r] A[i,k]) * (sum_l V[l*2+j, r] B[l,j])
    
    # Let's construct the 7 terms.
    # M1 = (A00+A11)(B00+B11)
    # U1: A00=1, A11=1. U1[0]=1, U1[3]=1.
    # V1: B00=1, B11=1. V1[0]=1, V1[3]=1.
    # W1: C00 += 1, C11 += 1. W1[0]=1, W1[3]=1.
    
    # M2 = (A10+A11)B00
    # U2: A10=1, A11=1. U2[2]=1, U2[3]=1.
    # V2: B00=1. V2[0]=1.
    # W2: C10 += 1, C11 += 1. W2[2]=1, W2[3]=1.
    
    # M3 = A00(B01-B11)
    # U3: A00=1. U3[0]=1.
    # V3: B01=1, B11=-1. V3[1]=1, V3[3]=-1.
    # W3: C01 += 1, C11 += 1. W3[1]=1, W3[3]=1.
    
    # M4 = A11(B10-B00)
    # U4: A11=1. U4[3]=1.
    # V4: B10=1, B00=-1. V4[2]=1, V4[0]=-1.
    # W4: C00 += 1, C10 += 1. W4[0]=1, W4[2]=1.
    
    # M5 = (A00+A01)B11
    # U5: A00=1, A01=1. U5[0]=1, U5[1]=1.
    # V5: B11=1. V5[3]=1.
    # W5: C00 -= 1, C01 += 1. W5[0]=-1, W5[1]=1.
    
    # M6 = (A10-A00)(B00+B01)
    # U6: A10=1, A00=-1. U6[2]=1, U6[0]=-1.
    # V6: B00=1, B01=1. V6[0]=1, V6[1]=1.
    # W6: C11 += 1. W6[3]=1.
    
    # M7 = (A01-A11)(B10+B11)
    # U7: A01=1, A11=-1. U7[1]=1, U7[3]=-1.
    # V7: B10=1, B11=1. V7[2]=1, V7[3]=1.
    # W7: C00 += 1, C11 += 1. W7[0]=1, W7[3]=1.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # r=2
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # r=3
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # r=4
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # r=5
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=6
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _construct_strassen_5x5():
    """
    Construct the rank-117 decomposition for 5x5x5.
    """
    n, m, p = 5, 5, 5
    total_rank = 117
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block sizes
    # Indices: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    # Block 0: {0,1}
    # Block 1: {2,3}
    # Block 2: {4}
    
    block_indices = [[0, 1], [2, 3], [4]]
    block_sizes = [2, 2, 1]
    
    col = 0
    
    # Get Strassen 2x2 matrices
    sU, sV, sW = _strassen_2x2()
    
    for i_blk in range(3):
        for k_blk in range(3):
            for j_blk in range(3):
                # A block: rows i_blk, cols k_blk
                # B block: rows k_blk, cols j_blk
                # C block: rows i_blk, cols j_blk
                
                size_i = block_sizes[i_blk]
                size_k = block_sizes[k_blk]
                size_j = block_sizes[j_blk]
                
                # Determine rank of this block multiplication
                if size_i == 2 and size_k == 2 and size_j == 2:
                    rank = 7
                    use_strassen = True
                else:
                    rank = size_i * size_k * size_j
                    use_strassen = False
                
                # Iterate through the local indices of the blocks
                if use_strassen:
                    # Map local Strassen indices to global
                    for r_local in range(7):
                        r_global = col + r_local
                        # U
                        for u_idx_local in range(4):
                            val = sU[u_idx_local][r_local]
                            if val != 0:
                                # Map u_idx_local (i*2+j) to global i, j
                                i_local = u_idx_local // 2
                                j_local = u_idx_local % 2
                                # Global i, j
                                i_global = block_indices[i_blk][i_local]
                                j_global = block_indices[k_blk][j_local]
                                U[i_global * m + j_global][r_global] += val
                        # V
                        for v_idx_local in range(4):
                            val = sV[v_idx_local][r_local]
                            if val != 0:
                                # Map v_idx_local (j*2+k) to global j, k
                                j_local = v_idx_local // 2
                                k_local = v_idx_local % 2
                                j_global = block_indices[k_blk][j_local]
                                k_global = block_indices[j_blk][k_local]
                                V[j_global * p + k_global][r_global] += val
                        # W
                        for w_idx_local in range(4):
                            val = sW[w_idx_local][r_local]
                            if val != 0:
                                # Map w_idx_local (k*2+i) to global k, i
                                k_local = w_idx_local // 2
                                i_local = w_idx_local % 2
                                k_global = block_indices[j_blk][k_local]
                                i_global = block_indices[i_blk][i_local]
                                W[k_global * n + i_global][r_global] += val
                else:
                    # Schoolbook for this block
                    for i_local in range(size_i):
                        for k_local in range(size_k):
                            for j_local in range(size_j):
                                r_global = col + (i_local * size_k * size_j) + (k_local * size_j) + j_local
                                i_global = block_indices[i_blk][i_local]
                                j_global = block_indices[k_blk][k_local] # A col
                                k_global = block_indices[j_blk][j_local] # B row? 
                                # Wait, A is i x k. B is k x j.
                                # A[i,k] * B[k,j] -> C[i,j].
                                # U index: i*m + k.
                                # V index: k*p + j.
                                # W index: j*n + i.
                                
                                U[i_global * m + j_global][r_global] += 1
                                V[j_global * p + k_global][r_global] += 1
                                W[k_global * n + i_global][r_global] += 1
                                # Note: In schoolbook, each term is 1.
                                # We add 1 to U, V, W.
                
                col += rank
    
    if col != total_rank:
        raise ValueError(f"Rank mismatch: expected {total_rank}, got {col}")
        
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition."""
    rank = len(U[0])
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for i in range(n):
        for j in range(p):
            for r in range(rank):
                f_r = np.dot(U_arr[i*m:(i+1)*m, r], A[i, :])
                g_r = np.dot(V_arr[(j*m):(j+1)*m, r], B[:, j]) # V index is k*p+j. 
                # V shape (m*p, rank). Index k*p+j.
                # For fixed j, V[:, r] is a vector of length m*p.
                # We need sum_k V[k*p+j, r] B[k,j].
                # This is dot product of B[:, j] with V[:, r] reshaped?
                # V[k*p+j] is the (k*p+j)-th element.
                # Let's extract the column for fixed j from V_arr.
                # V_col = V_arr.reshape(m, p, rank)[:, j, :]
                # Then dot(B[:, j], V_col[:, r])
                pass
    
    # Efficient verification
    # C[i,j] = sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] A[i,k]) * (sum_k V[k*p+j, r] B[k,j])
    # Note: W index in seed is W[k*n+i]. But in my Strassen construction I used W[j*n+i]?
    # Seed: W[k*n+i, r].
    # My Strassen: W[k*n+i, r].
    # In _strassen_2x2, W index is k*2+i.
    # In _construct_strassen_5x5, I used W[k_global * n + i_global].
    # So W is consistent.
    
    # Let's just do the loop, it's small enough (5x5x117).
    for i in range(n):
        for j in range(p):
            s = 0.0
            for r in range(rank):
                f_r = 0.0
                for k in range(m):
                    f_r += U[i*m+k][r] * A[i,k]
                g_r = 0.0
                for k in range(m):
                    g_r += V[k*p+j][r] * B[k,j]
                s += W[j*n+i][r] * f_r * g_r
            C_computed[i,j] = s
            
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Fallback
    fallback = _schoolbook(n, m, p)
    
    # Try Strassen 117
    if time.time() < deadline:
        try:
            result = _construct_strassen_5x5()
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
        except Exception as e:
            print(f"Strassen construction failed: {e}")
    
    return fallback

# EVOLVE-BLOCK-END
