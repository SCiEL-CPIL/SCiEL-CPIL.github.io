import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook solution as fallback
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
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    def to_half_int(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def check_exact(U, V, W, T, tol=1e-9):
        """Check if sum_c U[:,c] \otimes V[:,c] \otimes W[:,c] == T"""
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(recon, T, atol=tol)

    def try_rank_als(r, max_iters=100, seed=42):
        """
        Try to find a rank-r decomposition using ALS.
        We use a structured initialization based on the schoolbook pattern
        with random perturbations to break symmetry.
        """
        np.random.seed(seed)
        
        # Initialize with schoolbook-like structure but with r columns
        # We select r random (i,j,k) triples or use a structured subset
        # A better init: Random Gaussian, scaled appropriately
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: Fix V, W. Solve for U.
            # T_2d = U @ (V \otimes W)^T
            # (V \otimes W) has shape (r, m*p * p*n)
            # T_2d has shape (n*m, m*p * p*n)
            
            # Efficient computation of V \otimes W
            # V: (m*p, r), W: (p*n, r)
            # We need matrix B of shape (r, m*p*p*n) where B[c, b, d] = V[b, c] * W[d, c]
            # Then T_2d @ B^T gives U
            # B^T has shape (m*p*p*n, r)
            
            # Reshape T to (n*m, m*p*p*n)
            T_2d = T.reshape(n * m, -1)
            
            # Construct B^T efficiently
            # B^T is the transpose of the vectorized outer products
            # For each c, vec(V[:, c] * W[:, c])
            # This is expensive to build explicitly if r is large, but r ~ 50-60 is small.
            
            VW_2d = np.einsum('bc,dc->bdc', V, W).reshape(-1, r).T # Shape (r, m*p*p*n)
            # Actually, let's use the previous method which was clearer:
            # VW = np.einsum('bd,cd->bcd', V, W) -> shape (m*p, p*n, r)
            # VW_2d = VW.transpose(2, 0, 1).reshape(r, m*p*p*n)
            
            try:
                VW = np.einsum('bd,cd->bcd', V, W)
                VW_2d = VW.transpose(2, 0, 1).reshape(r, m * p * p * n)
                U_new = T_2d @ np.linalg.pinv(VW_2d.T)
                U = 0.5 * U + 0.5 * U_new
            except:
                pass
            
            # Update V
            T_2d_v = T.transpose(1, 0, 2).reshape(m * p, -1)
            try:
                UW = np.einsum('ad,cd->acd', U, W)
                UW_2d = UW.transpose(2, 0, 1).reshape(r, n * m * p * n)
                V_new = T_2d_v @ np.linalg.pinv(UW_2d.T)
                V = 0.5 * V + 0.5 * V_new
            except:
                pass
            
            # Update W
            T_2d_w = T.transpose(2, 0, 1).reshape(p * n, -1)
            try:
                UV = np.einsum('ad,bd->abd', U, V)
                UV_2d = UV.transpose(2, 0, 1).reshape(r, n * m * m * p)
                W_new = T_2d_w @ np.linalg.pinv(UV_2d.T)
                W = 0.5 * W + 0.5 * W_new
            except:
                pass
            
            # Check error
            recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-8:
                break
        
        # Try rounding to half-integers
        U_h = to_half_int(best_U)
        V_h = to_half_int(best_V)
        W_h = to_half_int(best_W)
        
        if check_exact(U_h, V_h, W_h, T):
            return U_h, V_h, W_h
            
        # Try integer rounding
        U_i = np.round(best_U)
        V_i = np.round(best_V)
        W_i = np.round(best_W)
        if check_exact(U_i, V_i, W_i, T):
            return U_i, V_i, W_i
            
        return None

    best_result = None
    
    # Theoretical lower bound is 20, but practical is likely 60.
    # We try a few ranks below 60 to see if we can find a decomposition.
    # Given the time constraint, we limit the search.
    for r in range(59, 55, -1):
        if time.time() > deadline - 20:
            break
        
        found = None
        # Fewer seeds for lower ranks as they are harder
        n_seeds = 5
        for seed in range(n_seeds):
            if time.time() > deadline - 20:
                break
            result = try_rank_als(r, max_iters=50, seed=seed * 100 + r)
            if result is not None:
                found = result
                break
        
        if found is not None:
            U, V, W = found
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
