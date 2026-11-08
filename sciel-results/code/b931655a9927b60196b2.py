import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We'll try to find a low-rank decomposition using randomized methods
    # First, let's set up the matrix multiplication tensor
    # T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # The tensor is indexed by (i,j) for U, (j,k) for V, (k,i) for W
    
    # Actually, let's think of this differently.
    # We want to decompose: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] such that C = sum_r U[:,r] * V[:,r] * W[:,r]
    # where U[i,j] corresponds to A[i,j], V[j,k] to B[j,k], W[k,i] to C[k,i]
    
    # Let's use a known result: for 3x3x5, the rank is 36 (from Strassen-type results)
    # But let's try to find it computationally
    
    # Create the multiplication tensor of shape (n*m, m*p, p*n)
    # T[(i,j), (j',k), (k',i')] = 1 if j==j' and k==k' and i==i'
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the tensor for matrix multiplication is:
    # T[a,b,c] where a = (i,j), b = (j,k), c = (k,i)
    # T[a,b,c] = 1 if the indices are consistent, i.e., the j in a matches j in b, 
    # the k in b matches k in c, and the i in c matches i in a
    
    # Let's enumerate:
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Create tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Now we want to find rank-r decomposition: T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
    # This means T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Let's try to find the minimal rank using iterative methods
    # Start with a reasonable upper bound and try to reduce
    
    # First, let's get a schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    # Verify schoolbook
    T_recon = np.einsum('ar,br,cr->abc', U_school, V_school, W_school)
    assert np.allclose(T_recon, T), "Schoolbook decomposition failed"
    
    # Now try to find a lower rank decomposition
    # Use a greedy approach: start with schoolbook, try to merge terms
    
    # Actually, let's try a different approach: use the fact that for 3x3x3, 
    # Strassen gives rank 7. For 3x3x5, we can pad or use block methods.
    
    # Let's try rank 36 first (known upper bound from literature)
    # But let's try to find it algorithmically
    
    # Try to use randomized SVD-like approach on the tensor
    # First, unfold the tensor to get matrices
    
    # T reshaped to (nm, mp*p*n)
    T_mat1 = T.reshape(nm, -1)
    T_mat2 = T.reshape(mp, -1)
    T_mat3 = T.reshape(pn, -1)
    
    # Get approximate ranks
    rank1 = np.linalg.matrix_rank(T_mat1)
    rank2 = np.linalg.matrix_rank(T_mat2)
    rank3 = np.linalg.matrix_rank(T_mat3)
    
    # The rank of the tensor is at least max(rank1, rank2, rank3)
    lower_bound = max(rank1, rank2, rank3)
    
    # For our case: 
    # T_mat1 has shape (9, 15*15=225), rank should be 9 (full)
    # T_mat2 has shape (15, 9*15=135), rank should be 15 (full)  
    # T_mat3 has shape (15, 9*15=135), rank should be 15 (full)
    # So lower bound is 15
    
    # Let's try to find a rank-15 to rank-36 decomposition
    # Use iterative compression
    
    def try_rank(r):
        """Try to find a rank-r decomposition"""
        if r < lower_bound:
            return None
        
        # Initialize with random factors
        np.random.seed(42)
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Iteratively optimize
        for iteration in range(500):
            # Update U: for each a, U[a,:] should minimize ||T[a,:,:] - sum_r U[a,r]*V[:,r]*W[:,r]||^2
            # = ||T[a,:,:].reshape(-1) - V * W^T * U[a,:]||^2 (approximately)
            # More precisely: minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # derivative wrt U[a,r'] = -2 * sum_{b,c} T[a,b,c] * V[b,r'] * W[c,r'] + 2 * sum_{r'} U[a,r'] * sum_{b,c} V[b,r']*V[b,r'] * W[c,r']*W[c,r']
            
            # Let's use a simpler iterative method: alternating least squares
            # Fix V, W, optimize U
            for a in range(nm):
                # Minimize ||T[a,:,:] - U[a,:] * (V*W^T) flattened||
                # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # = sum_r U[a,r] * (V[b,r] * W[c,r])
                # Let M[a] be matrix with M[a][b,c] = T[a,b,c]
                # Let N[b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # We want U[a,:] to best fit T[a,:,:]
                
                # This is a linear least squares: 
                # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # Flatten: T[a,:, :].reshape(-1) = U[a,:] @ (V*W^T).T.reshape(-1) ... 
                # Actually: sum_r U[a,r] * V[b,r] * W[c,r] = U[a,:] dot (V[b,:]*W[c,:])
                
                # Build design matrix: for each r, column r is V[:,r]*W[:,r] flattened
                # Design matrix D has shape (mp*pn, r)
                D = np.zeros((mp*pn, r))
                for rr in range(r):
                    D[:, rr] = (V[:, rr] * W[:, rr]).flatten()
                
                y = T[a, :, :].flatten()
                # Solve U[a,:] = (D^T D)^{-1} D^T y
                try:
                    U[a, :] = np.linalg.lstsq(D, y, rcond=None)[0]
                except:
                    U[a, :] = np.zeros(r)
            
            # Fix U, W, optimize V
            for b in range(mp):
                D = np.zeros((nm*pn, r))
                for rr in range(r):
                    D[:, rr] = (U[:, rr] * W[:, rr]).flatten()
                y = T[:, b, :].reshape(nm, -1).flatten()
                # Wait, T[:,b,:] has shape (nm, pn), flatten to (nm*pn)
                y = T[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(D, y, rcond=None)[0]
                except:
                    V[b, :] = np.zeros(r)
            
            # Fix U, V, optimize W
            for c in range(pn):
                D = np.zeros((nm*mp, r))
                for rr in range(r):
                    D[:, rr] = (U[:, rr] * V[:, rr]).flatten()
                y = T[:, :, c].reshape(nm, -1).flatten()
                y = T[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(D, y, rcond=None)[0]
                except:
                    W[c, :] = np.zeros(r)
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
            if error < 1e-10:
                break
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
        if error < 1e-8:
            # Convert to exact rational/half-integer if possible
            return U, V, W, error
        return None
    
    # Try ranks from lower bound upward
    best_result = None
    for r in range(lower_bound, rank_school + 1):
        result = try_rank(r)
        if result is not None:
            U, V, W, error = result
            best_result = (r, U, V, W)
            break
    
    if best_result is None:
        # Fall back to schoolbook
        r = rank_school
        U, V, W = U_school, V_school, W_school
    else:
        r, U, V, W = best_result
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            rlist = []
            for val in row:
                # Round to nearest half-integer
                half_int = round(val * 2) / 2
                rlist.append(half_int)
            result.append(rlist)
        return result
    
    U_list = to_list(U)
    V_list = to_list(V)
    W_list = to_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
