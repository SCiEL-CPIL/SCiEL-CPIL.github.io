import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
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
        """Round to nearest half-integer and check if valid."""
        rounded = np.round(arr * 2) / 2.0
        return rounded
    
    def check_exact(U, V, W, T, tol=1e-9):
        """Check if sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T"""
        # Compute the reconstructed tensor
        recon = np.zeros_like(T)
        for c in range(U.shape[1]):
            recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.allclose(recon, T, atol=tol)
    
    def try_rank(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using alternating least squares."""
        if r < max(n * m, m * p, p * n):
            return None  # Impossible
        
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each column c, minimize ||T - sum_c' U_c'⊗V_c'⊗W_c'||
            # This is: for each c, U[:,c] should approximate the "residual" projected
            # Actually, use ALS: fix V,W, solve for U; fix U,W, solve for V; fix U,V, solve for W
            
            # Update U: T reshaped as (n*m, m*p*p*n) vs V⊗W
            # T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
            # For fixed V,W: T_reshaped[a, (b,c)] = sum_d U[a,d] * (V[b,d]*W[c,d])
            # So: T_2d = U @ (V⊗W).T where V⊗W is (r, m*p*p*n)
            VW = np.einsum('bd,cd->bcd', V, W)  # (m*p, p*n, r)
            VW_2d = VW.transpose(2, 0, 1).reshape(r, m * p * p * n)
            T_2d = T.reshape(n * m, m * p * p * n)
            # U @ VW_2d.T = T_2d => U = T_2d @ pinv(VW_2d.T)
            try:
                U_new = T_2d @ np.linalg.pinv(VW_2d.T)
            except:
                U_new = U
            U = U_new
            
            # Update V
            UW = np.einsum('ad,cd->acd', U, W)  # (n*m, p*n, r)
            UW_2d = UW.transpose(2, 0, 1).reshape(r, n * m * p * n)
            T_2d_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V_new = T_2d_v @ np.linalg.pinv(UW_2d.T)
            except:
                V_new = V
            V = V_new
            
            # Update W
            UV = np.einsum('ad,bd->abd', U, V)  # (n*m, m*p, r)
            UV_2d = UV.transpose(2, 0, 1).reshape(r, n * m * m * p)
            T_2d_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W_new = T_2d_w @ np.linalg.pinv(UV_2d.T)
            except:
                W_new = W
            W = W_new
            
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
        
        # Try rounding the current (not just best)
        U_h2 = to_half_int(U)
        V_h2 = to_half_int(V)
        W_h2 = to_half_int(W)
        if check_exact(U_h2, V_h2, W_h2, T):
            return U_h2, V_h2, W_h2
        
        return None
    
    # Try ranks from schoolbook down
    best_result = None
    for r in range(rank_schoolbook, max(n * m, m * p, p * n) - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple seeds
        found = None
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = try_rank(r, max_iters=200, seed=seed + r)
            if result is not None:
                found = result
                break
        
        if found is not None:
            U, V, W = found
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
