import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Winograd rectangular fast matrix multiplication for 2x3 * 3x4
    # We use polynomial evaluation at specific points to construct rank-19 decomposition
    # 
    # Strategy: 
    # - For the 3-dimension (m=3), use evaluation at points {0, 1, -1} (3 points)
    # - For the 4-dimension (p=4), use evaluation at points {0, 1, -1, 2} (4 points)  
    # - For the 2-dimension (n=2), use evaluation at points {0, 1} (2 points)
    #
    # The tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    #
    # Winograd construction:
    # Let's define evaluation points for each dimension
    # For dimension of size 3: points alpha = [0, 1, -1]
    # For dimension of size 4: points beta = [0, 1, -1, 2]
    # For dimension of size 2: points gamma = [0, 1]
    
    # Actually, let me use a cleaner Winograd construction.
    # The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Represent A[i,j] as coefficients of polynomial a_i(x) = sum_j A[i,j] x^j (deg 2)
    # Represent B[j,k] as coefficients of polynomial b_k(y) = sum_j B[j,k] y^j (deg 2)
    # Then C[i,k] = sum_j A[i,j] B[j,k] is NOT simply a product of polynomials.
    #
    # Better approach: Use the standard Winograd for rectangular case.
    # For MxN * NxP where M=2, N=3, P=4:
    # 
    # We can write this as:
    # C = A * B where A is 2x3, B is 3x4
    # 
    # Winograd method: 
    # 1. Transform A: For each row i, compute D_A[i, t] = sum_j A[i,j] * u_t[j] for t in eval points
    # 2. Transform B: For each column k, compute D_B[k, s] = sum_j B[j,k] * v_s[j] for s in eval points
    # 3. Multiply: E[i, k, t, s] = D_A[i,t] * D_B[k,s] for valid (t,s) pairs
    # 4. Inverse transform to get C
    #
    # The number of scalar multiplications = number of valid (t,s) pairs
    #
    # For N=3: use 3 evaluation points, say {0, 1, -1}
    # The transform matrix U (3x3) has U[t,j] = (eval_point_t)^j
    # U = [[1, 0, 0], [1, 1, 1], [1, -1, 1]]
    # Inverse U_inv: we need U_inv * U = I
    #
    # For the 4-dimension in B's columns: we need to handle 4 columns.
    # Actually, B is 3x4, so we have 3 rows and 4 columns.
    # The "N" dimension is 3 (the shared dimension).
    #
    # Let me reconsider. The standard Winograd for MxN * NxP:
    # - M = 2, N = 3, P = 4
    # - We evaluate along the N dimension (size 3)
    # - Use r evaluation points where r < N for efficiency
    #
    # For N=3, using r=3 points gives the same as schoolbook (9 mults for the 3x3 part)
    # But we can be smarter with the rectangular shape.
    #
    # Let me just try to find a rank-19 decomposition using a numerical search
    # with the Winograd structure as guidance.
    
    start_time = time.time()
    max_time = 120.0
    
    # Target tensor
    T = np.ones((n, m, p), dtype=float)
    
    def verify(U, V, W):
        """Verify that U, V, W give exact decomposition of T."""
        K = U.shape[1]
        U_3d = U.reshape(n, m, K)
        V_3d = V.reshape(m, p, K)
        W_3d = W.reshape(p, n, K)
        try:
            T_recon = np.einsum('ijc,jpc,knc->ijk', U_3d, V_3d, W_3d, optimize=True)
        except Exception:
            return False
        return np.allclose(T_recon, T, atol=1e-9)
    
    def round_half(U, V, W):
        """Round to nearest half-integer."""
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2
    
    # Try to construct Winograd-style decomposition
    # For 2x3 * 3x4, a known result is rank 19.
    # 
    # Let me try a systematic construction:
    # The multiplication C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # We can think of this as: for each (i,k), we need sum_j A[i,j]B[j,k] = 1
    # in the tensor sense.
    #
    # Winograd approach for this specific case:
    # 
    # Let's use the following construction based on polynomial evaluation:
    # 
    # For the 3-dimension (j index), use evaluation at points {0, 1, -1}
    # For the 4-dimension (k index), we handle it directly
    # For the 2-dimension (i index), we handle it directly
    #
    # Actually, let me try a different approach: use the known fact that
    # the tensor rank of 2x3x4 matrix multiplication is 19, and construct
    # it using a parameterized family.
    
    # Let's try ALS (Alternating Least Squares) for rank 19
    def search_rank(K, max_restarts=100, max_iters=100):
        best = None
        for restart in range(max_restarts):
            if time.time() - start_time > max_time - 5:
                break
            
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, K)
            V = np.random.randn(m * p, K)
            W = np.random.randn(p * n, K)
            
            # Scale to help convergence
            U /= np.sqrt(n * m)
            V /= np.sqrt(m * p)  
            W /= np.sqrt(p * n)
            
            prev_loss = np.inf
            for it in range(max_iters):
                if time.time() - start_time > max_time - 5:
                    break
                
                # Update U: fix V, W
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For fixed (j,k), the contribution to U[i*m+j, :] is:
                # U[i*m+j, :] = argmin sum_k (T[i,j,k] - sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]))^2
                # This is a linear least squares for each (i,j)
                
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Build the design matrix for U[i*m+j, :]
                        # For each k, coefficient vector is V[j*p+k, :] * W[k*n+i, :]
                        # Shape: (p, K)
                        design = np.zeros((p, K))
                        for k in range(p):
                            design[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Target: T[i, j, :] which is all 1s of length p
                        target = T[i, j, :]  # shape (p,)
                        # Solve: design^T design x = design^T target
                        A_mat = design.T @ design  # (K, K)
                        b_vec = design.T @ target  # (K,)
                        try:
                            new_U[i * m + j, :] = np.linalg.solve(A_mat + 1e-8 * np.eye(K), b_vec)
                        except np.linalg.LinAlgError:
                            new_U[i * m + j, :] = np.linalg.lstsq(A_mat + 1e-8 * np.eye(K), b_vec, rcond=None)[0]
                
                U = new_U
                
                # Update V: fix U, W
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # For each i, coefficient vector is U[i*m+j, :] * W[k*n+i, :]
                        design = np.zeros((n, K))
                        for i in range(n):
                            design[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        target = T[:, j, k]  # shape (n,)
                        A_mat = design.T @ design
                        b_vec = design.T @ target
                        try:
                            new_V[j * p + k, :] = np.linalg.solve(A_mat + 1e-8 * np.eye(K), b_vec)
                        except np.linalg.LinAlgError:
                            new_V[j * p + k, :] = np.linalg.lstsq(A_mat + 1e-8 * np.eye(K), b_vec, rcond=None)[0]
                
                V = new_V
                
                # Update W: fix U, V
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # For each j, coefficient vector is U[i*m+j, :] * V[j*p+k, :]
                        design = np.zeros((m, K))
                        for j in range(m):
                            design[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        target = T[i, :, k]  # shape (m,)
                        A_mat = design.T @ design
                        b_vec = design.T @ target
                        try:
                            new_W[k * n + i, :] = np.linalg.solve(A_mat + 1e-8 * np.eye(K), b_vec)
                        except np.linalg.LinAlgError:
                            new_W[k * n + i, :] = np.linalg.lstsq(A_mat + 1e-8 * np.eye(K), b_vec, rcond=None)[0]
                
                W = new_W
                
                # Check loss
                if it % 5 == 0 or it == max_iters - 1:
                    loss = np.sum((T - np.einsum('ijc,jpc,knc->ijk', 
                                               U.reshape(n,m,K), V.reshape(m,p,K), 
                                               W.reshape(p,n,K), optimize=True))**2)
                    if loss < prev_loss * 0.999:
                        prev_loss = loss
                    if loss < 1e-12:
                        break
                
                if time.time() - start_time > max_time - 5:
                    break
            
            # Try rounding to half-integers
            U_r, V_r, W_r = round_half(U, V, W)
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
            
            # Also try without rounding in case we're close
            if verify(U, V, W):
                return U, V, W
        
        return None
    
    # Try rank 19 first (known optimal for 2x3x4)
    result = search_rank(19, max_restarts=50, max_iters=80)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 20
    if time.time() - start_time < max_time - 10:
        result = search_rank(20, max_restarts=30, max_iters=60)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 24
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

# EVOLVE-BLOCK-END
