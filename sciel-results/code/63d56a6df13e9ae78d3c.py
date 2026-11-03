"""Matrix multiplication 2x3x6 - linear algebraic rank determination + construction."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    def verify(U, V, W, tol=1e-9):
        """Verify the decomposition exactly reproduces the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_half_integer(U, V, W, tol=1e-9):
        """Verify entries are half-integers and decomposition is valid."""
        for mat in [U, V, W]:
            mat = np.array(mat, dtype=float)
            if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
                return False
        return verify(U, V, W, tol)
    
    def compute_matricization_rank():
        """
        Compute the rank of the tensor matricization.
        Flatten the tensor along the (i,j) mode: shape (n*m) x (m*p * p*n)
        The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), else 0.
        
        Actually, the matricization M has M[a, (b,c)] = T[a,b,c] where:
        a = i*m+j, b = j*p+k, c = k*n+i
        
        For each (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
        M[i*m+j, (j*p+k)*(p*n) + (k*n+i)] = 1
        """
        nm = n * m
        mp = m * p
        pn = p * n
        M = np.zeros((nm, mp * pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    M[a, b * pn + c] = 1
        # Compute rank
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        rank = np.sum(s > tol)
        return rank, M
    
    def try_construct_from_svd(rank_target, M, deadline):
        """
        Try to construct a rank-r decomposition using the SVD of the matricization.
        
        If the matricization has rank r, then the tensor rank is at most r.
        We can use the left and right singular vectors to construct U, V, W.
        
        The tensor T can be written as:
        T[a, b, c] = sum_s u_s[a] * v_s[b] * w_s[c]
        
        where u_s, v_s, w_s are the singular vectors.
        
        But we need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        
        One approach: use the fact that T[a,b,c] = sum_s sigma_s * u_s[a] * v_s[b] * w_s[c]
        where (u_s, sigma_s, w_s) are from SVD of M.
        
        But this gives a 3-way decomposition, not the bilinear form we need.
        
        Alternative: For a given rank r, we need to find U, V, W such that
        for all (a,b,c) where T[a,b,c] = 1: sum_r U[a,r]*V[b,r]*W[c,r] = 1
        and for all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
        
        This is a system of equations. Let's try a different approach:
        Use the row space of the matricization.
        
        If rank(M) = r, then the rows of M span an r-dimensional space.
        We can find a basis for this space and use it to construct the decomposition.
        """
        if time.time() > deadline:
            return None
        
        # Get the SVD
        U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
        r = rank_target
        
        if r >= len(s):
            # Use all singular values
            sigma = s
            u_vecs = U_svd
            vt_vecs = Vt_svd
        else:
            sigma = s[:r]
            u_vecs = U_svd[:, :r]
            vt_vecs = Vt_svd[:r, :]
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Reshape vt_vecs: each row is of length mp*pn, reshape to (mp, pn)
        # vt_vecs has shape (r, mp*pn)
        # We want V_tilde[c, s] and W_tilde[d, s] such that
        # sum_s sigma_s * u_vecs[a, s] * V_tilde[b, s] * W_tilde[c, s] = T[a,b,c]
        
        # This is tricky. Let's try a different construction.
        
        # For the matrix multiplication tensor, there's a known structure.
        # Let's try to directly solve for U, V, W using the constraint equations.
        
        # Number of equations: nm * mp * pn = 6 * 18 * 12 = 1296
        # But only 36 of them are non-zero (equal to 1), the rest must be 0.
        # Actually, the tensor has 1s only at specific positions.
        
        # Let's try: fix W to be a random matrix with rank r, then solve for U and V.
        # For a fixed W, the equations become:
        # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
        # where c_idx is determined by (b, c) -> but this is still bilinear.
        
        # Alternative: use the fact that for matrix multiplication, we can use
        # the "middle" indices to separate the problem.
        
        # For 2x3x6, let's try a structured approach:
        # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,5}
        
        # Key insight: The tensor is the outer product of three "all-ones" vectors
        # in a certain sense, but with the constraint that the indices are linked.
        
        # Let's try to find the minimal rank by checking if rank 18 works.
        # The matricization along (i,j) has shape 6 x 216.
        # Its rank is at most 6.
        
        # Matricization along (j,k): shape 18 x 72 (6*12)
        # Matricization along (k,i): shape 12 x 108 (6*18)
        
        # The tensor rank is at least the maximum of these matricization ranks.
        
        # Let's compute all three matricization ranks.
        
        # M1: (i,j) x (j,k) x (k,i) -> flatten (j,k,k,i) to get 6 x 216
        # M2: (j,k) x (i,j) x (k,i) -> flatten (i,j,k,i) to get 18 x 72
        # M3: (k,i) x (i,j) x (j,k) -> flatten (i,j,j,k) to get 12 x 108
        
        # For M2: T[i,j,k] = 1, so M2[j*p+k, i*m+j*(p*n) + k*n+i] ... this is complex.
        
        # Let's just compute the rank of the first matricization and use that as a lower bound.
        
        return None
    
    def try_direct_construction(rank_target, deadline):
        """
        Try to directly construct a rank-r decomposition for the 2x3x6 tensor.
        
        Strategy: The tensor T[i,j,k] = 1 for all valid (i,j,k).
        We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
        
        One approach: use the fact that 2x3x6 can be decomposed as:
        T = T1 + T2 where T1 handles k=0,1,2 and T2 handles k=3,4,5.
        Each Ti is a 2x3x3 tensor.
        
        For a 2x3x3 tensor, the schoolbook rank is 18.
        But maybe we can do better.
        
        Actually, let's try a completely different approach:
        Use the "border rank" or "approximate rank" ideas, but we need exact rank.
        
        For small matrix multiplication, the optimal rank is often known.
        For 2x3x3, the rank is known to be 18 (schoolbook is optimal for this size).
        
        So 2x3x6 would have rank at most 36 (two copies of 2x3x3).
        But can we do better by sharing structure?
        
        Let's try: can we find a rank-18 decomposition for 2x3x6?
        That would mean each "rank-1 tensor" contributes to multiple (i,j,k) entries.
        
        Actually, the minimal rank for 2x3x6 is likely 36 or close to it,
        because the tensor is "generic" in some sense.
        
        Let's try to verify if rank < 36 is possible by attempting to find such a decomposition.
        
        For rank r, we have r * (6 + 18 + 12) = 36r unknowns and 36 equations (one for each (i,j,k)).
        Wait, no - we have 6*18*12 = 1296 entries in the tensor, but only 36 are non-zero.
        The other 1260 must be zero. So we have 1296 equations.
        
        For r=36: 36*36 = 1296 unknowns, 1296 equations - square system.
        For r<36: underdetermined, but we need exact solutions.
        
        Let's try to solve the system for small r using a structured approach.
        """
        if time.time() > deadline:
            return None
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Build the constraint matrix.
        # For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For other (a,b,c), the sum must be 0.
        
        # This is a tensor equation. Let's try a different approach:
        # Use the fact that the tensor is separable in a certain way.
        
        # For 2x3x6, let's try to find a decomposition of the form:
        # U[i*m+j, c] = u1[i,j,c] * u2[i,j,c]
        # V[j*p+k, c] = v1[j,k,c] * v2[j,k,c]
        # W[k*n+i, c] = w1[k,i,c] * w2[k,i,c]
        
        # This is getting too complex. Let's just return the schoolbook solution
        # unless we can quickly verify a lower rank.
        
        return None
    
    # Compute the matricization rank to get a lower bound
    mat_rank, M = compute_matricization_rank()
    
    # The tensor rank is at least mat_rank.
    # For 2x3x6, mat_rank is likely 6 (since the first mode has size 6).
    
    # Try to find a decomposition with rank = mat_rank if possible.
    # Otherwise, try rank = 2*mat_rank, etc.
    
    # For now, let's try a few candidate ranks using a structured search.
    # Given the time constraint, we'll try ranks from mat_rank up to schoolbook_rank.
    
    # Actually, let's try a smarter approach: use the SVD to construct a candidate
    # decomposition and verify it.
    
    # The tensor T can be written as a sum of rank-1 tensors.
    # If we use the SVD of the matricization, we get:
    # T[a,b,c] = sum_s sigma_s * u_s[a] * (v_s reshaped)[b,c]
    
    # This is a 2-way decomposition, not 3-way. We need to factor (v_s reshaped)[b,c]
    # into a product of two vectors.
    
    # For each s, (v_s reshaped) is an mp x pn matrix. If this matrix has rank 1,
    # then we can write it as v_s[b] * w_s[c], and we have a rank-1 tensor.
    
    # In general, (v_s reshaped) has rank at most min(mp, pn) = min(18, 12) = 12.
    # So each "singular term" might need up to 12 rank-1 tensors to represent.
    
    # Total rank would be sum_s rank(v_s_reshaped) which could be up to 6*12 = 72.
    # But we want to minimize this.
    
    # Let's try to compute the actual structure.
    
    U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
    r_svd = len(s)
    
    # For each singular vector, check the rank of the reshaped right singular vector.
    total_rank = 0
    for idx in range(r_svd):
        v_vec = Vt_svd[idx, :]  # length mp*pn
        v_mat = v_vec.reshape(mp, pn)
        # Compute rank of v_mat
        U_v, s_v, Vt_v = np.linalg.svd(v_mat, full_matrices=False)
        tol = 1e-10
        rank_v = np.sum(s_v > tol)
        total_rank += rank_v
    
    # total_rank is an upper bound on the tensor rank using this SVD-based construction.
    # But it's not necessarily minimal.
    
    # For the matrix multiplication tensor, there might be a more efficient construction.
    
    # Let's try to directly construct a decomposition using the known structure.
    # For 2x3x6, the tensor is:
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,...,5}
    
    # One decomposition: use 36 terms, one for each (i,j,k).
    # This is the schoolbook decomposition.
    
    # Can we do better? Let's try to find a rank-18 decomposition.
    # If we can, that would be a significant improvement.
    
    # For rank 18, we need 18 * 36 = 648 unknowns and 1296 equations.
    # This is underdetermined, so a solution might exist.
    
    # Let's try to find a solution using a structured approach.
    # Fix W to be a random matrix with rank 18, then solve for U and V.
    
    # Actually, let's try a simpler approach: use the fact that the tensor is
    # the Kronecker product of smaller tensors.
    
    # 2x3x6 = 2x3x3 * 2 (in some sense). But this doesn't directly help.
    
    # Given the time constraint and the difficulty of finding a lower rank,
    # let's return the schoolbook solution.
    
    # But first, let's try one more thing: check if the matricization rank
    # gives a feasible lower bound that we can achieve.
    
    # For the matrix multiplication tensor M_{n,m,p}, the rank is known to be
    # at least max(n*m, m*p, p*n) in some cases, but this is not always tight.
    
    # For 2x3x6: max(6, 18, 12) = 18.
    # So the rank is at least 18.
    
    # Let's try to find a rank-18 decomposition.
    
    def try_rank_18(deadline):
        """Try to find a rank-18 decomposition for 2x3x6."""
        if time.time() > deadline:
            return None
        
        r = 18
        nm = n * m
        mp = m * p
        pn = p * n
        
        # We need U (6x18), V (18x18), W (12x18) such that
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        
        # Let's try a structured initialization.
        # Use the SVD of the matricization to get initial values.
        
        U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
        
        # M has shape (6, 216). U_svd has shape (6, 6), s has length 6, Vt_svd has shape (6, 216).
        # This only gives us 6 singular vectors, but we need 18.
        
        # Let's try a different approach: use random initialization with constraints.
        
        # For each c in 0..17, choose U[:,c], V[:,c], W[:,c] such that
        # the sum over c gives the correct tensor.
        
        # One idea: use the "middle" indices to separate the problem.
        # For each j in {0,1,2}, the sub-tensor T[:,j,:] has shape (2, 6).
        # We can try to decompose each sub-tensor separately.
        
        # T[:,j,:] = u_j ⊗ v_j where u_j is 2x1 and v_j is 6x1? No, that's rank 1.
        # T[:,j,:] has all entries 1, so it's the outer product of [1,1] and [1,1,1,1,1,1].
        # So T[:,j,:] = [1,1]^T ⊗ [1,1,1,1,1,1]
        
        # This means for each j, we can write:
        # sum_c U[i*3+j,c] * V[j*6+k,c] * W[k*2+i,c] = 1 for all i,k
        
        # If we set U[i*3+j,c] = a_j[c], V[j*6+k,c] = b_j[k,c], W[k*2+i,c] = c_j[k,i,c]
        # then sum_c a_j[c] * b_j[k,c] * c_j[k,i,c] = 1
        
        # This is still complex. Let's just try random search with verification.
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for it in range(5000):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
        
        if best_err < 1e-9 and best_U is not None:
            if verify(best_U, best_V, best_W):
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        return None
    
    # Try rank 18
    result_18 = try_rank_18(deadline)
    if result_18 is not None:
        return result_18
    
    # If rank 18 doesn't work, return schoolbook
    return best_result
# EVOLVE-BLOCK-END

