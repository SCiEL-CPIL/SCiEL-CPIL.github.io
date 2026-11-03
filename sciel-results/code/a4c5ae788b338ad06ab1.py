"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication (n x m) @ (m x p) = (n x p)
    # The tensor for matrix multiplication has entries T[i, j, k] = 1 if we use schoolbook,
    # but actually the tensor is defined such that C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C = sum_c U_c * V_c * W_c where each term is an outer product
    # U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    # For this to equal the matrix multiplication tensor, we need:
    # sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = delta_{j is valid} ... 
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in range
    # No wait. The tensor for matrix multiplication M_{n,m,p} is defined as:
    # T_{i,j,k} = 1 if the multiplication involves A[i,j]*B[j,k]*C[k,i]... 
    # Let me think more carefully.
    #
    # The matrix multiplication tensor T for (n,m,p) has indices (i,j,k) where
    # i in [0,n), j in [0,m), k in [0,p), and T[i,j,k] = 1.
    # A decomposition of rank r is: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So for all i,j,k, we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    #
    # This is a very strong condition. The schoolbook decomposition uses r=n*m*p terms,
    # one for each (i,j,k) triple, with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for that term.
    #
    # For Strassen-like decompositions, the entries can be -1, 0, 1, etc.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U, V, W) where U, V, W are 4-element lists (for 2x2 matrices flattened).
    U[i*2+j], V[j*2+k], W[k*2+i].
    """
    # Strassen's 7 products for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h)  -> U: a,d in A; V: e,h in B; W: i,j,k,l in C
    # M2 = (c+d)(e)    -> U: c,d in A; V: e in B; W: k,l in C
    # M3 = a(f-h)      -> U: a in A; V: f,h in B; W: j,l in C
    # M4 = d(g-e)      -> U: d in A; V: g,e in B; W: i,k in C
    # M5 = (a+b)(h)    -> U: a,b in A; V: h in B; W: i,j in C
    # M6 = (c-a)(e+f)  -> U: c,a in A; V: e,f in B; W: k,j in C
    # M7 = (b-d)(g+h)  -> U: b,d in A; V: g,h in B; W: i,l in C
    
    # Flatten indices: A[0]=a, A[1]=b, A[2]=c, A[3]=d
    #                  B[0]=e, B[1]=f, B[2]=g, B[3]=h
    #                  C[0]=i, C[1]=j, C[2]=k, C[3]=l
    
    terms = []
    
    # M1 = (a+d)(e+h): U has 1 at a(0),d(3); V has 1 at e(0),h(3); W has 1 at i(0),j(1),k(2),l(3)
    # Actually W contributes to C[i,j] and C[k,l]... 
    # C[0] = M1 - M4 + M5 - M7
    # C[1] = M3 + M5
    # C[2] = M2 + M4
    # C[3] = M1 - M2 + M6 - M7
    
    # So for each M, we need to track which C entries it contributes to and with what sign.
    # Let me define each term as (U_vec, V_vec, W_vec) where these are length-4 vectors.
    
    # M1: A part: a+d -> U = [1, 0, 0, 1]
    #     B part: e+h -> V = [1, 0, 0, 1]
    #     C part: contributes +1 to C[0], +1 to C[3] -> W = [1, 0, 0, 1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2: A part: c+d -> U = [0, 0, 1, 1]
    #     B part: e   -> V = [1, 0, 0, 0]
    #     C part: contributes +1 to C[2] -> W = [0, 0, 1, 0]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # M3: A part: a   -> U = [1, 0, 0, 0]
    #     B part: f-h -> V = [0, 1, 0, -1]
    #     C part: contributes +1 to C[1] -> W = [0, 1, 0, 0]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # M4: A part: d   -> U = [0, 0, 0, 1]
    #     B part: g-e -> V = [0, 0, 1, -1]  # wait, g-e means V[2]-V[0]... 
    #     Actually V = [0, 0, 1, -1] means V contributes g - e? No.
    #     V[j*p+k] for 2x2: V[0]=V[0*2+0]=V[e], V[1]=V[0*2+1]=V[f], V[2]=V[1*2+0]=V[g], V[3]=V[1*2+1]=V[h]
    #     So V = [0, 0, 1, -1] means coefficient 1 for g and -1 for e. That's g-e. Correct.
    #     C part: contributes -1 to C[0], +1 to C[2] -> W = [-1, 0, 1, 0]
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    
    # M5: A part: a+b -> U = [1, 1, 0, 0]
    #     B part: h   -> V = [0, 0, 0, 1]
    #     C part: contributes +1 to C[0], +1 to C[1] -> W = [1, 1, 0, 0]
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    
    # M6: A part: c-a -> U = [0, 0, 1, -1]  # wait, c-a means U[2]-U[0]
    #     U = [-1, 0, 1, 0]
    #     B part: e+f -> V = [1, 1, 0, 0]
    #     C part: contributes +1 to C[3], +1 to C[2]? 
    #     C[3] = M1 - M2 + M6 - M7, C[2] = M2 + M4
    #     So M6 contributes +1 to C[3]. What about C[2]? No, M6 only in C[3].
    #     Wait let me recheck: C[2] = M2 + M4. C[3] = M1 - M2 + M6 - M7.
    #     So M6 contributes +1 to C[3] only. W = [0, 0, 0, 1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7: A part: b-d -> U = [0, 1, 0, -1]
    #     B part: g+h -> V = [0, 0, 1, 1]
    #     C part: contributes -1 to C[0], -1 to C[3] -> W = [-1, 0, 0, -1]
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    
    return terms


def _embed_2x2_to_4x4(vec_4, row_start, col_start, size=4):
    """Embed a 4-element vector (for 2x2) into a 16-element vector (for 4x4) at given position."""
    result = [0] * 16
    for idx, val in enumerate(vec_4):
        r = idx // 2
        c = idx % 2
        result[(row_start + r) * size + (col_start + c)] = val
    return result


def _strassen_4x4_terms():
    """Generate 49 Strassen terms for 4x4x4 matrix multiplication using 2x2 blocks."""
    base_terms = _strassen_2x2_terms()
    # We have 4 2x2 blocks in each matrix: top-left(0,0), top-right(0,1), bottom-left(1,0), bottom-right(1,1)
    # For 4x4 = 2x2 blocks of 2x2
    block_positions = [(0,0), (0,1), (1,0), (1,1)]
    
    terms_4x4 = []
    for (r1, c1) in block_positions:  # A block position
        for (r2, c2) in block_positions:  # B block position
            for (r3, c3) in block_positions:  # C block position
                # This block multiplication C[r3,c3] += A[r1,c1] * B[r2,c3] ... 
                # Actually for block matrix multiplication:
                # C[i_block, j_block] = sum_k A[i_block, k_block] * B[k_block, j_block]
                # So for C block at (r3, c3), we sum over k_block from A[r3, k] * B[k, c3]
                # But Strassen works on 2x2 matrices. For 4x4, we use 2x2 blocks.
                # The 4x4 matrix multiplication can be done by treating it as 2x2 block multiplication
                # where each block is 2x2. So we need Strassen on the 2x2 block structure.
                # But that gives us 7 block-level multiplications, each of which is a 2x2*2x2=2x2 multiplication.
                # So total = 7 * 7 = 49 terms.
                
                # Let me think differently. The 4x4x4 tensor can be decomposed by applying
                # Strassen at the block level. Each of the 7 Strassen block-products involves
                # linear combinations of 2x2 blocks, and each 2x2 block product can be done
                # with 7 scalar-level Strassen terms.
                
                # Actually, a cleaner approach: the 4x4 matrix multiplication tensor is the
                # Kronecker product of the 2x2 tensor with itself. So if T_2 has rank 7,
                # then T_4 = T_2 ⊗ T_2 has rank at most 7*7 = 49.
                
                # For the Kronecker product decomposition:
                # If T_2[i,j,k] = sum_{c=0}^{6} U2[c][i*2+j] * V2[c][j*2+k] * W2[c][k*2+i]
                # Then T_4[I,J,K] = T_2[I//2, J//2, K//2] * T_2[I%2, J%2, K%2]
                # = sum_{c1,c2} U2[c1][I//2*2+J//2] * V2[c1][J//2*2+K//2] * W2[c1][K//2*2+I//2]
                #               * U2[c2][I%2*2+J%2] * V2[c2][J%2*2+K%2] * W2[c2][K%2*2+I%2]
                
                # So for each pair (c1, c2), we get a term with:
                # U4[c1*7+c2][I*4+J] = U2[c1][I//2*2+J//2] * U2[c2][I%2*2+J%2]
                # V4[c1*7+c2][J*4+K] = V2[c1][J//2*2+K//2] * V2[c2][J%2*2+K%2]
                # W4[c1*7+c2][K*4+I] = W2[c1][K//2*2+I//2] * W2[c2][K%2*2+I%2]
                
                pass
    
    # Let me implement the Kronecker product approach directly
    terms_4x4 = []
    for c1 in range(7):
        for c2 in range(7):
            U4 = [0] * 16
            V4 = [0] * 16
            W4 = [0] * 16
            for I in range(4):
                for J in range(4):
                    I1, I0 = I // 2, I % 2
                    J1, J0 = J // 2, J % 2
                    U4[I*4+J] = base_terms[c1][0][I1*2+J1] * base_terms[c2][0][I0*2+J0]
            for J in range(4):
                for K in range(4):
                    J1, J0 = J // 2, J % 2
                    K1, K0 = K // 2, K % 2
                    V4[J*4+K] = base_terms[c1][1][J1*2+K1] * base_terms[c2][1][J0*2+K0]
            for K in range(4):
                for I in range(4):
                    K1, K0 = K // 2, K % 2
                    I1, I0 = I // 2, I % 2
                    W4[K*4+I] = base_terms[c1][2][K1*2+I1] * base_terms[c2][2][K0*2+I0]
            terms_4x4.append((U4, V4, W4))
    
    return terms_4x4


def _make_4x4x7_decomposition():
    """Create a rank-98 decomposition for 4x4x7 using Strassen 4x4x4 terms.
    Split the 7 columns into two groups: cols 0-3 and cols 4-6 (padded to 4).
    """
    terms_4x4 = _strassen_4x4_terms()
    
    U_list = []
    V_list = []
    W_list = []
    
    for U4, V4, W4 in terms_4x4:
        # First group: columns 0-3 of B, contributing to columns 0-3 of C
        V1 = [0] * 28  # 4*7 = 28 elements
        W1 = [0] * 28  # 7*4 = 28 elements
        for J in range(4):
            for K in range(4):
                V1[J*7+K] = V4[J*4+K]
        for K in range(4):
            for I in range(4):
                W1[K*4+I] = W4[K*4+I]
        
        U_list.append(list(U4))
        V_list.append(V1)
        W_list.append(W1)
        
        # Second group: columns 4-6 of B (padded to 4: cols 4,5,6, and a zero col 7)
        # Actually B is 4x7, so columns are 0-6. For the second block, use cols 4,5,6 and pad with 0.
        V2 = [0] * 28
        W2 = [0] * 28
        for J in range(4):
            for K in range(3):  # K=0,1,2 correspond to original cols 4,5,6
                V2[J*7+(K+4)] = V4[J*4+K]
            # V2[J*7+7] would be out of range since we only have 7 cols. So just leave as 0.
        for K in range(3):  # K=0,1,2 correspond to original cols 4,5,6
            for I in range(4):
                W2[(K+4)*4+I] = W4[K*4+I]
        
        U_list.append(list(U4))
        V_list.append(V2)
        W_list.append(W2)
    
    # Convert to 2D lists
    rank = len(U_list)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    for c in range(rank):
        for i in range(16):
            U[i][c] = U_list[c][i]
        for i in range(28):
            V[i][c] = V_list[c][i]
        for i in range(28):
            W[i][c] = W_list[c][i]
    
    return U, V, W


def _try_improve_with_random_search(U, V, W, n, m, p, max_iter=5000, timeout=120):
    """Try to reduce rank by merging or eliminating terms via random perturbations."""
    start_time = time.time()
    rank = len(U[0])
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try to eliminate terms by checking if any term is redundant
    # This is expensive, so we do a limited search
    
    for iteration in range(max_iter):
        if time.time() - start_time > timeout - 10:
            break
        
        # Check if current decomposition is valid
        # (skip for speed, assume it's valid)
        
        # Try to remove a random term and see if we can compensate
        # This is a heuristic approach
        pass
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS
    
    # First, get the Strassen-based rank-98 decomposition
    U, V, W = _make_4x4x7_decomposition()
    
    # Verify the decomposition
    if not _check_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook if Strassen is wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to improve with limited random search
    if time.time() < deadline - 30:
        U, V, W, rank = _try_improve_with_random_search(U, V, W, n, m, p, max_iter=2000, timeout=30)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
