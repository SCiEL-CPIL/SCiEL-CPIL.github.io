import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Build rank-28 baseline
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))
    
    col = 0
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # 7 terms for 2x2x2 Strassen
        # M1 = (A11 + A22)(B11 + B22)
        U28[i1*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        # M2 = A22(B21 - B11)
        U28[i2*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1
        W28[k1*n+i2, col] = 1; W28[k2*n+i2, col] = -1
        col += 1
        # M3 = (A12 - A22)B22
        U28[i1*m+j1, col] = 1
        V28[j1*p+k2, col] = 1; V28[j2*p+k2, col] = -1
        W28[k2*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        # M4 = A11(B12 - B22)
        U28[i2*m+j2, col] = 1
        V28[j2*p+k1, col] = 1; V28[j1*p+k1, col] = -1
        W28[k1*n+i1, col] = 1; W28[k1*n+i2, col] = 1
        col += 1
        # M5 = (A11 + A12)B11
        U28[i1*m+j1, col] = 1; U28[i1*m+j2, col] = 1
        V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = -1; W28[k2*n+i1, col] = 1
        col += 1
        # M6 = (A21 - A11)(B11 + B12)
        U28[i2*m+j1, col] = 1; U28[i1*m+j1, col] = -1
        V28[j1*p+k1, col] = 1; V28[j1*p+k2, col] = 1
        W28[k2*n+i2, col] = 1
        col += 1
        # M7 = (A12 - A22)(B21 + B22)
        U28[i1*m+j2, col] = 1; U28[i2*m+j2, col] = -1
        V28[j2*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U28[i*m+j, col] = 1
                    V28[j*p+k, col] = 1
                    W28[k*n+i, col] = 1
                    col += 1

    # Apply Strassen to (i=0,1; j=0,1; k=0,1)
    add_strassen(0, 1, 0, 1, 0, 1)
    # Apply Strassen to (i=0,1; j=0,1; k=2,3)
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Schoolbook for remaining parts involving j=2 or k=4
    # A[:,0:2] * B[0:2, 4:5] -> C[:, 4:5]
    add_schoolbook([0, 1], [0, 1], [4])
    # A[:,0:2] * B[2:3, 0:2] -> C[:, 0:2]
    add_schoolbook([0, 1], [2], [0, 1])
    # A[:,0:2] * B[2:3, 2:4] -> C[:, 2:4]
    add_schoolbook([0, 1], [2], [2, 3])
    # A[:,0:2] * B[2:3, 4:5] -> C[:, 4:5]
    add_schoolbook([0, 1], [2], [4])

    # Verify rank-28
    def verify_decomposition(U, V, W, rank):
        res = np.zeros((n, m, p))
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        target = np.ones((n, m, p))
        return np.allclose(res, target, atol=1e-9)

    if not verify_decomposition(U28, V28, W28, rank_28):
        # Fallback to schoolbook rank 30 if something went wrong
        U30 = np.zeros((n * m, 30))
        V30 = np.zeros((m * p, 30))
        W30 = np.zeros((p * n, 30))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U30[i*m+j, col] = 1
                    V30[j*p+k, col] = 1
                    W30[k*n+i, col] = 1
                    col += 1
        return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}

    # Now try to reduce rank from 28 to 27 using randomized linear algebra
    # The idea: if there's a linear dependency among the 28 factors, we can eliminate one.
    # Specifically, we look for coefficients c_0, ..., c_27 (not all zero) such that
    # sum_c c_c * (U_c ⊗ V_c ⊗ W_c) = 0
    # If such a dependency exists, we can reparameterize to get rank 27.
    
    # Build the tensor and check its rank via random projections
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want to find if rank(T) < 28
    
    # Use random linear combinations to detect dependencies
    # Generate random vectors in the factor spaces and check if the resulting
    # tensor has rank < 28
    
    np.random.seed(42)
    
    # Try multiple random linear combinations
    best_rank = 28
    best_factors = None
    
    num_trials = 200
    trial_budget = 60.0  # seconds for rank reduction search
    
    for trial in range(num_trials):
        if time.time() > start_time + trial_budget:
            break
            
        # Generate random coefficients
        coeffs = np.random.randn(rank_28)
        coeffs /= np.linalg.norm(coeffs)
        
        # Compute the "effective" factors: U_eff = U28 @ coeffs, etc.
        # But we need to check if the 28 factors span a space of dimension < 28
        
        # Alternative approach: check if the 28 rank-1 tensors are linearly dependent
        # by flattening them and computing the rank of the matrix with these as columns
        
        # Flatten each factor triple into a vector of length n*m*p
        flat_factors = np.zeros((n*m*p, rank_28))
        for c in range(rank_28):
            Uc = U28[:, c].reshape(n, m)
            Vc = V28[:, c].reshape(m, p)
            Wc = W28[:, c].reshape(p, n)
            # The tensor contribution is Uc[i,j] * Vc[j,k] * Wc[k,i]
            # Flatten in order (i,j,k)
            tensor_c = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_c[i, j, k] = Uc[i, j] * Vc[j, k] * Wc[k, i]
            flat_factors[:, c] = tensor_c.flatten()
        
        # Compute the rank of flat_factors
        # If rank < 28, there's a linear dependency
        try:
            rank_ff = np.linalg.matrix_rank(flat_factors, tol=1e-10)
        except Exception:
            rank_ff = rank_28
        
        if rank_ff < best_rank:
            best_rank = rank_ff
            # Find the null space and use it to construct a lower-rank decomposition
            try:
                null_vec = np.linalg.svd(flat_factors, full_matrices=True)[2][-1]
                # null_vec gives coefficients such that sum_c null_vec[c] * factor_c = 0
                # We can eliminate one factor using this dependency
                # For now, just record that a lower rank is possible
            except Exception:
                pass
    
    # For now, if we detected rank < 28, we could attempt to construct the decomposition
    # But constructing it correctly requires more work. Let's just return the rank-28 for now
    # unless we can actually construct a rank-27 decomposition.
    
    # Actually, let's try a more direct approach: use the null space to eliminate a factor
    if best_rank < 28:
        try:
            # Recompute flat_factors
            flat_factors = np.zeros((n*m*p, rank_28))
            for c in range(rank_28):
                Uc = U28[:, c].reshape(n, m)
                Vc = V28[:, c].reshape(m, p)
                Wc = W28[:, c].reshape(p, n)
                tensor_c = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor_c[i, j, k] = Uc[i, j] * Vc[j, k] * Wc[k, i]
                flat_factors[:, c] = tensor_c.flatten()
            
            # Get null space
            U_sv, s_vals, Vt = np.linalg.svd(flat_factors, full_matrices=True)
            # The last row of Vt is the null vector
            null_vec = Vt[-1]
            
            # Find the index with the largest absolute coefficient in null_vec
            max_idx = np.argmax(np.abs(null_vec))
            max_coeff = null_vec[max_idx]
            
            if abs(max_coeff) > 1e-8:
                # We can express factor max_idx as a combination of the others
                # factor_max_idx = -sum_{c != max_idx} (null_vec[c] / max_coeff) * factor_c
                # So we can eliminate factor max_idx
                
                # Construct new factors of rank 27
                new_rank = 27
                U_new = np.zeros((n * m, new_rank))
                V_new = np.zeros((m * p, new_rank))
                W_new = np.zeros((p * n, new_rank))
                
                # Copy all factors except max_idx
                new_col = 0
                for c in range(rank_28):
                    if c == max_idx:
                        continue
                    U_new[:, new_col] = U28[:, c]
                    V_new[:, new_col] = V28[:, c]
                    W_new[:, new_col] = W28[:, c]
                    new_col += 1
                
                # Verify this works
                if verify_decomposition(U_new, V_new, W_new, new_rank):
                    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                else:
                    # The elimination didn't work (the dependency might not be simple)
                    # Fall through to return rank-28
                    pass
        except Exception:
            pass
    
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
