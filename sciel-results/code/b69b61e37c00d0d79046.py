import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a decomposition with rank < 120 using ALS
    # The tensor T has indices (i, j, j', k) with T[i, j, j', k] = 1 if j == j' else 0.
    # We want sum_c W[k, i, c] * U[i, j, c] * V[j', k, c] = T[i, j, j', k]
    
    # Reshape indices for easier vectorization
    # U shape: (n, m, R)
    # V shape: (m, p, R)
    # W shape: (p, n, R)
    
    def run_als(R, iters=50, seed=42):
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n, m, R) * 0.1
        V = np.random.randn(m, p, R) * 0.1
        W = np.random.randn(p, n, R) * 0.1
        
        # Target tensor T[i, j, j', k] = delta(j, j')
        # We can compute the error efficiently without storing the full T if it's large,
        # but n*m*m*p = 4*5*5*6 = 600 elements. It's small.
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, j, k] = 1.0
        
        for _ in range(iters):
            # Update U
            # For each i, j, c:
            # b[j', k] = T[i, j, j', k] - sum_{c' != c} W[k, i, c'] U[i, j, c'] V[j', k, c']
            # X[j', k] = W[k, i, c] V[j', k, c]
            # U[i, j, c] = dot(b, X) / dot(X, X)
            
            # We can compute the "rest" term (sum over all c) and subtract the c-th term
            # Sum_c W[k, i, c] U[i, j, c] V[j', k, c]
            # Let S[i, j, j', k] = sum_c ...
            
            # Efficient update for U:
            # Fix V, W.
            # Compute current sum S = einsum('kic,ijc,j'kc->ij'k', W, U, V) ? No, indices are tricky.
            # W is (p, n, R), U is (n, m, R), V is (m, p, R).
            # Term c: W[k, i, c] * U[i, j, c] * V[j', k, c].
            # Sum over c: S[i, j, j', k] = sum_c W[k, i, c] U[i, j, c] V[j', k, c].
            
            # We want to update U[i, j, c].
            # Let A_c[i, j, j', k] = W[k, i, c] V[j', k, c].
            # Then S[i, j, j', k] = sum_c U[i, j, c] A_c[i, j, j', k].
            # Note A_c depends on i, j, j', k.
            
            # For a fixed i, j, the vector U[i, j, :] is multiplied by the tensor A[i, j, :, :].
            # This is a linear system for each (i, j).
            # But we want to update one c at a time? No, we can update all c for a fixed (i, j) simultaneously?
            # No, the equations are coupled in c.
            # However, if we use the "one factor at a time" ALS, we fix V and W, and solve for all U.
            # This is a linear least squares problem for each (i, j) if we treat U[i, j, :] as the variable vector.
            
            # Let y[i, j, j', k] = T[i, j, j', k].
            # We want to approximate y with sum_c U[i, j, c] A_c[i, j, j', k].
            # This is equivalent to:
            # For each i, j:
            #   Let Y = T[i, j, :, :] (shape m x p)
            #   Let A_mat be a matrix of shape (m*p, R) where A_mat[idx, c] = W[k, i, c] V[j', k, c] with idx = j'*p+k.
            #   Solve min_U || Y.flatten() - A_mat @ U[i, j, :] ||^2.
            
            for i in range(n):
                for j in range(m):
                    # Construct A_mat for this (i, j)
                    # A_mat has columns c.
                    # Element (j', k) of column c is W[k, i, c] * V[j', k, c].
                    # Shape of A_mat: (m*p, R)
                    A_mat = np.zeros((m * p, R))
                    for c in range(R):
                        # W[:, i, c] is shape (p,)
                        # V[:, :, c] is shape (m, p)
                        # We need to multiply element-wise?
                        # Term is W[k, i, c] * V[j', k, c].
                        # This is an outer product of W[:, i, c] (shape p) and V[:, :, c] (shape m, p)?
                        # No, V[j', k, c].
                        # For fixed k, W[k, i, c] is a scalar. It multiplies the row V[:, k, c] of V?
                        # No, V is (m, p, R). V[j', k, c].
                        # So for fixed k, the vector over j' is V[:, k, c].
                        # So A_mat[:, c] is the vectorization of sum_k (W[k, i, c] * V[:, k, c])?
                        # No, the indices j' and k are separate in the output tensor.
                        # The output is indexed by (j', k).
                        # So A_mat is not a simple sum.
                        # A_mat[j'*p + k, c] = W[k, i, c] * V[j', k, c].
                        
                        # We can compute this as:
                        # Temp[k, j'] = W[k, i, c] * V[j', k, c]
                        # This is an outer product of W[:, i, c] (shape p) and V.T[:, :, c] (shape p, m)?
                        # V[j', k, c]. Let's transpose V to (p, m, R).
                        # Then V_T[k, j', c] = V[j', k, c].
                        # Temp[k, j'] = W[k, i, c] * V_T[k, j', c].
                        # This is element-wise multiplication of W[:, i, c] (shape p, 1) and V_T[:, :, c] (shape p, m).
                        # Result shape (p, m).
                        # We need to flatten to (m*p).
                        
                        pass # We will compute this in a loop for clarity/speed trade-off, or use einsum.
                    
                    # Using einsum is better.
                    # A_mat_flat = np.zeros((m*p, R))
                    # for c in range(R):
                    #    # W_k_ic = W[:, i, c] (p,)
                    #    # V_jk_c = V[:, :, c] (m, p)
                    #    # We want M[j', k] = W[k, i, c] * V[j', k, c]
                    #    # This is V[:, :, c] * W[:, i, c][None, :]
                    #    M = V[:, :, c] * W[:, i, c][None, :] # Shape (m, p)
                    #    A_mat_flat[:, c] = M.flatten()
                    
                    A_mat_flat = np.zeros((m * p, R))
                    for c in range(R):
                        M = V[:, :, c] * W[:, i, c][None, :] # (m, p)
                        A_mat_flat[:, c] = M.flatten()
                    
                    y = T[i, j, :, :].flatten() # (m*p,)
                    
                    # Solve least squares: A_mat_flat @ u = y
                    # u = U[i, j, :]
                    if R > 0:
                        U[i, j, :], residuals, rank, sv = np.linalg.lstsq(A_mat_flat, y, rcond=None)
            
            # Update V
            # Similar logic. Fix U, W.
            # For each j', k:
            #   y = T[:, j', j', k] ? No.
            #   T[i, j, j', k] = delta(j, j').
            #   For fixed j', k, the target vector over i is T[i, 0, j', k] ... no.
            #   The term is sum_c U[i, j, c] V[j', k, c] W[k, i, c].
            #   For fixed j', k, we want to find V[j', k, :].
            #   Let B_mat have shape (n*m, R).
            #   B_mat[i*m+j, c] = U[i, j, c] * W[k, i, c].
            #   Target y[i*m+j] = T[i, j, j', k].
            
            for j_prime in range(m):
                for k in range(p):
                    B_mat_flat = np.zeros((n * m, R))
                    for c in range(R):
                        # U[i, j, c] shape (n, m)
                        # W[k, i, c] shape (n,)
                        # M[i, j] = U[i, j, c] * W[k, i, c]
                        M = U[:, :, c] * W[k, :, c][:, None] # (n, m)
                        B_mat_flat[:, c] = M.flatten()
                    
                    # Target T[i, j, j', k]
                    y = T[:, :, j_prime, k].flatten() # (n*m,)
                    
                    if R > 0:
                        V[j_prime, k, :], residuals, rank, sv = np.linalg.lstsq(B_mat_flat, y, rcond=None)
            
            # Update W
            # For each k, i:
            #   y = T[i, :, :, k] ? No.
            #   T[i, j, j', k].
            #   For fixed k, i, we want W[k, i, :].
            #   Let C_mat have shape (m*m, R).
            #   C_mat[j*m+j', c] = U[i, j, c] * V[j', k, c].
            #   Target y[j*m+j'] = T[i, j, j', k].
            
            for k in range(p):
                for i in range(n):
                    C_mat_flat = np.zeros((m * m, R))
                    for c in range(R):
                        # U[i, :, c] shape (m,)
                        # V[:, k, c] shape (m,)
                        # M[j, j'] = U[i, j, c] * V[j', k, c]
                        # Outer product
                        M = np.outer(U[i, :, c], V[:, k, c]) # (m, m)
                        C_mat_flat[:, c] = M.flatten()
                    
                    y = T[i, :, :, k].flatten() # (m*m,)
                    
                    if R > 0:
                        W[k, i, :], residuals, rank, sv = np.linalg.lstsq(C_mat_flat, y, rcond=None)
            
            # Check error
            # Compute S = sum_c W[k, i, c] U[i, j, c] V[j', k, c]
            # S[i, j, j', k]
            S = np.zeros((n, m, m, p))
            for c in range(R):
                # Term = W[:, :, c] (p, n) * U[:, :, c] (n, m) * V[:, :, c] (m, p)
                # We need to combine indices.
                # T_c[i, j, j', k] = W[k, i, c] U[i, j, c] V[j', k, c]
                # This is a tensor product.
                # We can compute this via einsum or loops.
                # Since n,m,p are small, loops are fine.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Factor = W[k, i, c] * U[i, j, c]
                            factor = W[k, i, c] * U[i, j, c]
                            # Add factor * V[:, k, c] to S[i, j, :, k]
                            S[i, j, :, k] += factor * V[:, k, c]
            
            err = np.linalg.norm(S - T)
            if err < 1e-8:
                return U, V, W, True
        
        return U, V, W, False

    # Try ranks from 110 down to 100
    for R in range(110, 99, -1):
        U, V, W, success = run_als(R, iters=20)
        if success:
            # Convert to lists
            U_list = [[0]*R for _ in range(n*m)]
            V_list = [[0]*R for _ in range(m*p)]
            W_list = [[0]*R for _ in range(p*n)]
            
            for i in range(n):
                for j in range(m):
                    U_list[i*m+j] = U[i, j, :].tolist()
            for j in range(m):
                for k in range(p):
                    V_list[j*p+k] = V[j, k, :].tolist()
            for k in range(p):
                for i in range(n):
                    W_list[k*n+i] = W[k, i, :].tolist()
            
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
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
