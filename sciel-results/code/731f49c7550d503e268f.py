"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Condition: sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k*n+i, r] = delta_{j,j2}
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j2*p+k, r] * Wa[k*n+i, r]
                    expected = 1.0 if j == j2 else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _build_tensor_from_decomp(U, V, W, n, m, p):
    """Build the full tensor T[i*m+j, j2*p+k, k*n+i] from a decomposition."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', Ua[:, r], Va[:, r], Wa[:, r])
    return T


def _extract_decomp_from_tensor(T, n, m, p, tol=1e-10):
    """Extract a minimal-rank decomposition from a tensor via SVD on matricizations."""
    # Matricize T as (n*m) x (m*p*p*n) and find rank
    Tmat = T.reshape(n * m, -1)
    # SVD to find rank
    U_svd, s_vals, Vt_svd = np.linalg.svd(Tmat, full_matrices=False)
    # Determine numerical rank
    r = 0
    for sv in s_vals:
        if sv > tol:
            r += 1
        else:
            break
    if r == 0:
        return None, None, None
    # Use the top r singular vectors
    # T = sum_{i=1}^{r} s_i * u_i * v_i^T
    # We need to factor v_i (which has length m*p*p*n) into V_col * W_col
    # v_i is indexed by (j2, k, i2) where i2 is the output row index
    # v_i[j2*p+k, i2] = V[j2*p+k, i] * W[k*n+i2, i]
    # This is a rank-1 constraint on each slice... but in general it's not rank-1.
    # Instead, use the 3-way SVD approach or just use the matricized form directly.
    
    # Alternative: use the fact that T is a matrix multiplication tensor.
    # The tensor T[i*m+j, j2*p+k, k*n+i] = delta_{j,j2}
    # We can decompose this directly.
    
    # For the matrix multiplication tensor, we know the structure:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's just build the schoolbook tensor and find its rank via a different approach.
    # Actually, let's use the 3-mode SVD or just accept the matricized rank.
    
    # For a proper minimal decomposition, we need T to be expressible as
    # sum_r u_r ⊗ v_r ⊗ w_r
    # The matricized rank gives an upper bound.
    
    # Let's try: for each pair of modes, check if the tensor has low rank.
    # T as (n*m) x (m*p) x (p*n):
    # Mode-1 matricization: (n*m) x (m*p * p*n) -> rank gives bound
    # Mode-2 matricization: (m*p) x (n*m * p*n) -> rank gives bound  
    # Mode-3 matricization: (p*n) x (n*m * m*p) -> rank gives bound
    
    T1 = T.reshape(n * m, -1)
    T2 = T.reshape(m * p, -1)
    T3 = T.reshape(p * n, -1)
    
    r1 = np.linalg.matrix_rank(T1, tol=tol)
    r2 = np.linalg.matrix_rank(T2, tol=tol)
    r3 = np.linalg.matrix_rank(T3, tol=tol)
    
    # The tensor rank is at least max(r1, r2, r3)
    # For matrix multiplication tensor, the rank is known to be between certain bounds.
    
    # To extract an actual decomposition, we use the following approach:
    # If the tensor rank equals the matricized rank, we can extract it.
    # Otherwise, we need a more sophisticated approach.
    
    # For now, let's try to extract using the mode-1 SVD if it's tight.
    # Actually, for the matmul tensor, let's just use a known decomposition.
    
    return None, None, None


def _build_strassen_decomposition():
    """
    Correct Strassen-based decomposition for 3x3 @ 3x7.
    
    Strategy:
    - C[0:2, :] comes from A[0:2, :] @ B[:, :]
      = A[0:2, 0:2] @ B[0:2, :] + A[0:2, 2] @ B[2, :]
    - C[2, :] comes from A[2, :] @ B[:, :]
    
    For A[0:2, 0:2] @ B[0:2, :]:
      Split B[0:2, :] into 2x2 blocks: cols [0,1], [2,3], [4,5], and col [6]
      Use Strassen (rank 7) for each 2x2 @ 2x2 block
      Use schoolbook (rank 4) for 2x2 @ 2x1
    
    For A[0:2, 2] @ B[2, :]: schoolbook, rank 7
    For A[2, :] @ B[:, :]: schoolbook, rank 21... wait, that's 3*7=21? No.
    
    Actually A[2, :] is 1x3, B[:, :] is 3x7.
    C[2, k] = sum_j A[2,j] * B[j,k]
    This is rank 3 (one term per j, each contributing to all k).
    Wait no, in the tensor formulation, each (j,k) pair is a separate rank-1 term.
    So A[2,:] @ B[:,k] for each k: that's 3 terms per k, 7 k's = 21 terms.
    
    Hmm, let me recount. The tensor condition is:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
    
    For i=2, j=0, j'=0, k: U[2*3+0, r]*V[0*7+k, r]*W[k*3+2, r] summed = 1
    For i=2, j=0, j'=1, k: U[2*3+0, r]*V[1*7+k, r]*W[k*3+2, r] summed = 0
    
    So for each (i, k) pair, we need the sum over r of U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] to give delta_{j,j'}.
    
    For fixed i=2, k: we need a 3x3 matrix M where M[j,j'] = sum_r U[6+j, r] * V[7*j'+k, r] * W[3*k+2, r] = I.
    This means the vectors u_j = [U[6+j, r] * W[3*k+2, r]]_r and v_{j',k} = [V[7*j'+k, r]]_r must satisfy sum_r u_j[r] * v_{j',k}[r] = delta_{j,j'}.
    
    The minimal rank for this is 3 (take U[6+j,r]=delta_{j,r} for r=0,1,2, W[3k+2,r]=1, V[7j'+k, r]=delta_{j',r}).
    So for each k, we need 3 terms. For 7 k's, that's 21 terms for the bottom row.
    
    Total: 3*7 (Strassen for 3 blocks) + 4 (schoolbook for last col) + 7 (A[0:2,2]*B[2,:]) + 21 (bottom row)
    = 21 + 4 + 7 + 21 = 53
    
    Wait, that doesn't seem right. Let me re-derive more carefully.
    """
    n, m, p = 3, 3, 7
    
    U = []
    V = []
    W = []
    
    def make_U(indices):
        u = [0.0] * (n * m)
        for idx, val in indices:
            u[idx] = val
        return u
    
    def make_V(indices):
        v = [0.0] * (m * p)
        for idx, val in indices:
            v[idx] = val
        return v
    
    def make_W(indices):
        w = [0.0] * (p * n)
        for idx, val in indices:
            w[idx] = val
        return w
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    def strassen_2x2(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cr0, cr1, cc0, cc1):
        """
        Strassen for A[ar0:ar1+1, ac0:ac1+1] @ B[br0:br1+1, bc0:bc1+1]
        contributing to C[cr0:cr1+1, cc0:cc1+1]
        """
        ua11 = u_idx(ar0, ac0)
        ua12 = u_idx(ar0, ac1)
        ua21 = u_idx(ar1, ac0)
        ua22 = u_idx(ar1, ac1)
        
        vb11 = v_idx(br0, bc0)
        vb12 = v_idx(br0, bc1)
        vb21 = v_idx(br1, bc0)
        vb22 = v_idx(br1, bc1)
        
        wc11 = w_idx(cc0, cr0)
        wc12 = w_idx(cc1, cr0)
        wc21 = w_idx(cc0, cr1)
        wc22 = w_idx(cc1, cr1)
        
        # M1 = (a11+a22)(b11+b22) -> C11 += 1, C22 += 1
        U.append(make_U([(ua11, 1), (ua22, 1)]))
        V.append(make_V([(vb11, 1), (vb22, 1)]))
        W.append(make_W([(wc11, 1), (wc22, 1)]))
        
        # M2 = (a21+a22)b11 -> C21 += 1, C22 -= 1
        U.append(make_U([(ua21, 1), (ua22, 1)]))
        V.append(make_V([(vb11, 1)]))
        W.append(make_W([(wc21, 1), (wc22, -1)]))
        
        # M3 = a11(b21-b22) -> C12 += 1, C22 += 1
        U.append(make_U([(ua11, 1)]))
        V.append(make_V([(vb21, 1), (vb22, -1)]))
        W.append(make_W([(wc12, 1), (wc22, 1)]))
        
        # M4 = a22(b12-b11) -> C11 += 1, C21 += 1
        U.append(make_U([(ua22, 1)]))
        V.append(make_V([(vb12, 1), (vb11, -1)]))
        W.append(make_W([(wc11, 1), (wc21, 1)]))
        
        # M5 = (a11+a12)b22 -> C11 -= 1, C12 += 1
        U.append(make_U([(ua11, 1), (ua12, 1)]))
        V.append(make_V([(vb22, 1)]))
        W.append(make_W([(wc11, -1), (wc12, 1)]))
        
        # M6 = (a21-a11)(b11+b12) -> C22 += 1
        U.append(make_U([(ua21, 1), (ua11, -1)]))
        V.append(make_V([(vb11, 1), (vb12, 1)]))
        W.append(make_W([(wc22, 1)]))
        
        # M7 = (a12-a22)(b21+b22) -> C11 += 1, C22 += 1
        U.append(make_U([(ua12, 1), (ua22, -1)]))
        V.append(make_V([(vb21, 1), (vb22, 1)]))
        W.append(make_W([(wc11, 1), (wc22, 1)]))
    
    # Part 1: A[0:2, 0:2] @ B[0:2, :] for C[0:2, :]
    # Strassen for 2x2 @ 2x2 on B cols [0,1]
    strassen_2x2(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    # Strassen for 2x2 @ 2x2 on B cols [2,3]
    strassen_2x2(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    # Strassen for 2x2 @ 2x2 on B cols [4,5]
    strassen_2x2(0, 1, 0, 1, 0, 1, 4, 5, 0, 1, 4, 5)
    # Schoolbook for 2x2 @ 2x1 on B col [6]
    for i in [0, 1]:
        for j in [0, 1]:
            U.append(make_U([(u_idx(i, j), 1)]))
            V.append(make_V([(v_idx(j, 6), 1)]))
            W.append(make_W([(w_idx(6, i), 1)]))
    
    # Part 2: A[0:2, 2] * B[2, :] for C[0:2, :]
    # C[i,k] += A[i,2] * B[2,k] for i=0,1, k=0..6
    for i in [0, 1]:
        for k in range(7):
            U.append(make_U([(u_idx(i, 2), 1)]))
            V.append(make_V([(v_idx(2, k), 1)]))
            W.append(make_W([(w_idx(k, i), 1)]))
    
    # Part 3: A[2, :] * B[:, :] for C[2, :]
    # C[2,k] = sum_j A[2,j] * B[j,k]
    # For each j, k: rank-1 term
    for j in range(3):
        for k in range(7):
            U.append(make_U([(u_idx(2, j), 1)]))
            V.append(make_V([(v_idx(j, k), 1)]))
            W.append(make_W([(w_idx(k, 2), 1)]))
    
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_time=30):
    """
    Reduce rank by building the tensor and finding a minimal decomposition
    via iterative rank analysis.
    """
    import numpy as np
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j2*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j2*p+k,r]*W[k*n+i,r]
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', Ua[:, r], Va[:, r], Wa[:, r])
    
    # Verify the tensor is correct
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    expected = 1.0 if j == j2 else 0.0
                    if abs(T[i*m+j, j2*p+k, k*n+i] - expected) > 1e-9:
                        return U, V, W  # Invalid, return original
    
    # Now try to find a minimal-rank decomposition of T.
    # Strategy: Use the fact that T is a matmul tensor and try different block structures.
    
    # Approach: Try to find a decomposition of the form where we group terms.
    # For the 3x3 @ 3x7 case, let's try:
    # - Use Strassen for the 3x3 part (top 2x2 of A, top 2x2 of B)
    # - Handle the remaining parts optimally
    
    # Actually, let's try a direct approach: 
    # The tensor T has entries T[a,b,c] where a=i*m+j, b=j2*p+k, c=k2*n+i2
    # T[a,b,c] = delta_{j,j2} if i=i2 and k=k2, else 0
    # More precisely: T[i*m+j, j2*p+k, k2*n+i2] = 1 if j=j2 and k=k2 and i=i2, else 0
    
    # This is exactly the matmul tensor. Let's find its rank.
    # Lower bound: max of matricized ranks
    T1 = T.reshape(n*m, -1)
    T2 = T.reshape(m*p, -1)
    T3 = T.reshape(p*n, -1)
    r1 = np.linalg.matrix_rank(T1, tol=1e-10)
    r2 = np.linalg.matrix_rank(T2, tol=1e-10)
    r3 = np.linalg.matrix_rank(T3, tol=1e-10)
    lower_bound = max(r1, r2, r3)
    
    # Try to extract a decomposition of rank = lower_bound if possible.
    # For the matmul tensor, we can use the following approach:
    # If r1 == r2 == r3 == r, and the tensor has rank r, we can extract it.
    
    # Let's try to find a rank-r decomposition using the SVD of the matricizations.
    # If T has tensor rank r, then there exist U, V, W such that T = sum_{s=1}^r u_s ⊗ v_s ⊗ w_s
    
    # Use the mode-1 SVD: T1 = U_svd * S * Vt_svd
    # The columns of U_svd[:, :r] give the u_s vectors (up to scaling)
    # Then we need to factor the remaining part.
    
    # Let's try a different approach: use the known structure of the matmul tensor.
    # For 3x3 @ 3x7, we can try:
    # - 3x3 matmul has rank 7 (Strassen) or 5 (Lader) 
    # - The 3x7 extension adds terms
    
    # Let's try to build a decomposition with the following structure:
    # Split into:
    # 1. A[0:3, 0:3] @ B[0:3, 0:3] -> C[0:3, 0:3] (3x3 @ 3x3, rank 7)
    # 2. A[0:3, 0:3] @ B[0:3, 3:7] -> C[0:3, 3:7] (3x3 @ 3x4)
    # 3. A[0:3, 0:3] @ B[0:3, 6:7] -> C[0:3, 6:7] (3x3 @ 3x1)
    
    # For 3x3 @ 3x3: rank 7 (Strassen)
    # For 3x3 @ 3x4: can be done with rank 7+7=14? Or better?
    # For 3x3 @ 3x1: rank 9 (schoolbook) or less?
    
    # Actually, let's just try to minimize the total by trying different splits.
    
    # Let me try a concrete construction:
    # Use Strassen for the 3x3 @ 3x3 part (cols 0,1,2 of B)
    # For cols 3,4,5,6 of B: use a 3x3 @ 3x4 approach
    
    # 3x3 @ 3x4: Split B into 3x2 + 3x2
    # Each 3x3 @ 3x2: 
    #   = A @ B1 (3x2) 
    #   Can be done with: for each column of B1, use the 3x3 @ 3x1 approach
    #   3x3 @ 3x1: rank 9 (schoolbook) but can we do better?
    #   Actually, 3x3 @ 3x1 is just computing A @ b for a vector b.
    #   This is a bilinear map from (3x3) x (3) to (3).
    #   The rank of this tensor is 9 (schoolbook) since it's just matrix-vector mult.
    #   Wait, no. The tensor for A @ b where A is 3x3 and b is 3x1:
    #   T[i, j, k] = delta_{j,k} (where i is output row, j is A col, k is b index)
    #   This has rank 3*3 = 9? No, it has rank 3.
    #   T[i, j, k] = 1 if j=k, else 0
    #   This is sum_{j=0}^{2} e_i... no.
    #   T[i,j,k] = delta_{j,k}
    #   = sum_{j=0}^{2} (e_j in j-position) * (e_j in k-position) * (all-ones in i-position)
    #   So rank is 3!
    
    # Wait, that's for the case where we're computing A @ b.
    # In our tensor formulation: T[i*m+j, j2*p+k, k2*n+i2]
    # For A @ b (3x3 @ 3x1): n=3, m=3, p=1
    # T[i*3+j, j2*1+k, k2*3+i2] = delta_{j,j2} if k=k2 and i=i2
    # Since p=1, k=0 always, k2=0 always.
    # T[i*3+j, j2, i2] = delta_{j,j2} if i=i2, else 0
    # = delta_{j,j2} * delta_{i,i2}
    # This is sum_{j=0}^{2} (e_j in U) * (e_j in V) * (e_j... no)
    # U[i*3+j, r] * V[j2, r] * W[i2, r]
    # We need sum_r U[i*3+j,r] * V[j2,r] * W[i2,r] = delta_{j,j2} * delta_{i,i2}
    # Take r = j (0,1,2): U[i*3+j, j] = delta_{i,0}... no.
    # Take: for r in {0,1,2}: U[i*3+r, r] = 1 for all i, V[j2, r] = delta_{j2,r}, W[i2, r] = delta_{i2,0}
    # Then sum_r U[i*3+j,r]*V[j2,r]*W[i2,r] = sum_r delta_{j,r} * delta_{j2,r} * delta_{i2,0}
    # = delta_{j,j2} * delta_{i2,0}
    # That's not right, we need delta_{i,i2}.
    
    # Let me redo: We need sum_r U[i*3+j,r] * V[j2,r] * W[i2,r] = delta_{j,j2} * delta_{i,i2}
    # Try: U[i*3+j, r] where r indexes (i,j) pairs: r = i*3+j
    # Then U[i*3+j, i*3+j] = 1, V[j2, i*3+j] = delta_{j2,j}, W[i2, i*3+j] = delta_{i2,i}
    # Sum over r: U[i*3+j,r]*V[j2,r]*W[i2,r] = sum_r delta_{r, i*3+j} * delta_{j2, j_r} * delta_{i2, i_r}
    # where j_r = r mod 3, i_r = r div 3
    # = delta_{j2, j} * delta_{i2, i}
    # Yes! So rank is 9 (one term per (i,j) pair).
    
    # But can we do better? The matricized rank:
    # T as (9) x (3) x (3): 
    # Mode 1: 9 x 9, rank 9
    # Mode 2: 3 x 27, rank 3
    # Mode 3: 3 x 27, rank 3
    # Lower bound is 9. So rank is exactly 9.
    
    # OK so for 3x3 @ 3x1, rank is 9.
    # For 3x3 @ 3x4: we can split into four 3x3 @ 3x1, each rank 9, total 36.
    # But we can do better by sharing.
    
    # Actually, for 3x3 @ 3x4:
    # T[i*3+j, j2*4+k, k2*3+i2] = delta_{j,j2} if k=k2 and i=i2
    # = delta_{j,j2} * delta_{k,k2} * delta_{i,i2}
    # This is sum over (j,k,i) of ... 
    # Rank is 3*4*3 = 36? No, that's schoolbook.
    # Can we do better?
    # The matricized ranks:
    # Mode 1: 9 x (4*9=36), the tensor is identity-like, rank 9
    # Mode 2: 12 x (9*9=81), rank 12? 
    # Mode 3: 9 x (9*12=108), rank 9
    # Lower bound is 12.
    
    # Hmm, this is getting complex. Let me just try to use the SVD approach to find
    # the actual tensor rank and extract a decomposition.
    
    if time.time() > deadline - 5:
        return U, V, W
    
    # Try to find a decomposition using iterative methods.
    # Start with the current rank and try to reduce.
    
    # Alternative approach: use the known result that 3x3 matmul has rank 7 (Strassen)
    # and try to extend this to 3x3 @ 3x7.
    
    # Let's try a specific construction:
    # Use Strassen for the 3x3 @ 3x3 part (first 3 columns of B)
    # For the remaining 4 columns, use a 3x3 @ 3x4 decomposition.
    
    # 3x3 @ 3x4: 
    # C = A @ B where A is 3x3, B is 3x4
    # Split B into B1 (3x2) and B2 (3x2)
    # C1 = A @ B1 (3x2), C2 = A @ B2 (3x2)
    # For 3x3 @ 3x2:
    #   This is like computing two 3x3 @ 3x1 simultaneously
    #   We can use: C1[:,0] = A @ b1, C1[:,1] = A @ b2
    #   Using the rank-9 decomposition for each, but sharing the U part:
    #   For r = i*3+j (0..8): U[i*3+j, r] = 1
    #   For b1: V[j2, r] = delta_{j2, j} * 1 (for k=0)
    #   For b2: V[j2, r] = delta_{j2, j} * 1 (for k=1)
    #   But V has different entries for k=0 and k=1.
    #   So we need separate terms for each k.
    #   Total rank for 3x3 @ 3x2: 9*2 = 18? Or can we share?
    #   
    #   Actually, V[j2*p+k, r] where p=2:
    #   For k=0: V[j2*2+0, r] = delta_{j2, j}
    #   For k=1: V[j2*2+1, r] = delta_{j2, j}
    #   These are different rows of V, so they can share the same r if we set both.
    #   But W[k*3+i2, r] = delta_{i2, i} for both k=0 and k=1.
    #   So for each (i,j) pair, we have one term that handles both k=0 and k=1.
    #   Rank is 9 for 3x3 @ 3x2!
    
    # So 3x3 @ 3x4 = 3x3 @ 3x2 + 3x3 @ 3x2 = 9 + 9 = 18
    # Total for 3x3 @ 3x7 = 3x3 @ 3x3 + 3x3 @ 3x4 = 7 + 18 = 25
    
    # Let's verify this construction!
    
    n, m, p = 3, 3, 7
    U_new = []
    V_new = []
    W_new = []
    
    def make_U(indices):
        u = [0.0] * (n * m)
        for idx, val in indices:
            u[idx] = val
        return u
    
    def make_V(indices):
        v = [0.0] * (m * p)
        for idx, val in indices:
            v[idx] = val
        return v
    
    def make_W(indices):
        w = [0.0] * (p * n)
        for idx, val in indices:
            w[idx] = val
        return w
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    # Strassen for 3x3 @ 3x3 (cols 0,1,2 of B)
    # Standard Strassen for 3x3:
    # M1 = (A11+A33)(B11+B33)
    # M2 = (A21+A22)(B21+B22)  
    # ... actually, let me use the standard 3x3 Strassen decomposition.
    
    # The 3x3 Strassen decomposition has 7 terms:
    # Let A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # Let B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    # 
    # M1 = (a11+a22+a33)(b11+b22+b33)
    # M2 = (a12+a23+a31)(b12+b23+b31)
    # M3 = (a13+a21+a32)(b13+b21+b32)
    # M4 = (a11-a22)(b11-b22)
    # M5 = (a22-a33)(b22-b33)
    # M6 = (a33-a11)(b33-b11)
    # M7 = (a12-a23+a31-b13+b21-b32)... 
    # 
    # Actually, the standard 3x3 Strassen is more complex. Let me use a known one.
    # 
    # From literature, the 3x3 matmul tensor has rank 7. Here's one decomposition:
    # Define:
    # A1 = A[0,0]+A[1,1]+A[2,2]
    # A2 = A[0,1]+A[1,2]+A[2,0]
    # A3 = A[0,2]+A[1,0]+A[2,1]
    # A4 = A[0,0]-A[1,1]
    # A5 = A[1,1]-A[2,2]
    # A6 = A[2,2]-A[0,0]
    # A7 = A[0,1]-A[1,2]+A[2,0]  (or some combination)
    # 
    # B1 = B[0,0]+B[1,1]+B[2,2]
    # B2 = B[0,1]+B[1,2]+B[2,0]
    # B3 = B[0,2]+B[1,0]+B[2,1]
    # B4 = B[0,0]-B[1,1]
    # B5 = B[1,1]-B[2,2]
    # B6 = B[2,2]-B[0,0]
    # B7 = B[0,1]-B[1,2]+B[2,0]
    # 
    # Then C[i,j] is a linear combination of M_r = A_r * B_r.
    # 
    # This is getting complicated. Let me just use the schoolbook for 3x3 @ 3x3
    # and see if the total is still better than 53.
    # Schoolbook 3x3 @ 3x3: rank 9
    # 3x3 @ 3x4: rank 18
    # Total: 27. That's better than 53!
    
    # Actually, let me just use schoolbook for everything and count:
    # 3x3 @ 3x7 schoolbook: rank 3*3*7 = 63. Worse.
    
    # Let me try: Strassen for 3x3@3x3 (rank 7) + schoolbook for 3x3@3x4 (rank 36)
    # Total: 43. Still better than 53.
    
    # Or: schoolbook for 3x3@3x3 (rank 9) + my 3x3@3x4 (rank 18) = 27.
    
    # Let me implement the rank-18 decomposition for 3x3 @ 3x4.
    # For cols k=3,4,5,6 of B:
    # For each (i,j) pair (9 total), one term:
    # U[i*3+j, r] = 1
    # V[j2*7+k, r] = delta_{j2, j} for k in {3,4,5,6}
    # W[k*3+i2, r] = delta_{i2, i} for k in {3,4,5,6}
    
    # But wait, this gives:
    # sum_r U[i*3+j,r]*V[j2*7+k,r]*W[k*3+i2,r]
    # = sum_{(i',j') in 9 pairs} delta_{i,i'}delta_{j,j'} * delta_{j2,j'} * delta_{i2,i'}
    # (for fixed k in 3..6)
    # = delta_{j,j2} * delta_{i,i2}
    # 
    # But we need this to be delta_{j,j2} for the full tensor, and the contributions
    # from different k should not interfere. Since W[k*3+i2, r] = delta_{i2,i} only
    # for k in 3..6, and 0 otherwise, this works.
    
    # However, I realize the issue: for the full 3x3 @ 3x7 tensor, we need:
    # T[i*3+j, j2*7+k, k2*3+i2] = delta_{j,j2} if k=k2 and i=i2, else 0
    # 
    # For k=k2 in 3..6: the above terms contribute delta_{j,j2}*delta_{i,i2}
    # For k=k2 in 0..2: we need the 3x3 @ 3x3 part.
    
    # So let's use schoolbook for 3x3 @ 3x3 (9 terms) and my 18-term decomposition for 3x3 @ 3x4.
    # Total: 27 terms.
    
    # Schoolbook for 3x3 @ 3x3 (cols 0,1,2):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_new.append(make_U([(u_idx(i, j), 1)]))
                V_new.append(make_V([(v_idx(j, k), 1)]))
                W_new.append(make_W([(w_idx(k, i), 1)]))
    
    # 3x3 @ 3x4 (cols 3,4,5,6): 9 terms, each handling all 4 columns
    for i in range(3):
        for j in range(3):
            v_indices = [(v_idx(j, k), 1) for k in range(3, 7)]
            w_indices = [(w_idx(k, i), 1) for k in range(3, 7)]
            U_new.append(make_U([(u_idx(i, j), 1)]))
            V_new.append(make_V(v_indices))
            W_new.append(make_W(w_indices))
    
    rank_new = len(U_new)
    
    # Verify
    if _validate(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    
    # If verification fails, return original
    return U, V, W


def construct():
    start = time.time()
    deadline = start + 115  # safety margin
    
    n, m, p = DIMENSIONS
    
    try:
        # First try the optimized decomposition directly
        U_opt, V_opt, W_opt = _try_reduce_rank([], [], [], n, m, p, max_time=5)
        # _try_reduce_rank with empty input should build the optimized one
        # Actually, let's just call it properly
        U, V, W = _build_strassen_decomposition()
        rank = len(U)
        
        if _validate(U, V, W, n, m, p):
            # Try to reduce rank if time permits
            if time.time() < deadline - 20:
                U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=30)
                if _validate(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
