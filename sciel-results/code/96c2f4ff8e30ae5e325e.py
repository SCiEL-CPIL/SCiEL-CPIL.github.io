"""Rank-minimal exact tensor decomposition for 3x3 by 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 130  # safety margin before 150s kill

    def try_rank(r, max_restarts=30, max_iters=3000):
        """Try to find a rank-r decomposition using alternating least squares."""
        if time.time() > deadline:
            return None
        
        nm = n * m  # 9
        mp = m * p  # 12
        np_ = p * n  # 12
        
        # Build the tensor T as a 3D array
        T = np.zeros((nm, mp, np_))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        best_err = float('inf')
        best_solution = None
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random values
            np.random.seed(42 + attempt * 1000)
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(np_, r) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each row a, solve least squares
                # T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # Vectorize: T[a].reshape(-1) = sum_r U[a,r] * (V[:,r] * W[:,r])
                # So we need: for each a, solve [V_r * W_r for r] @ U[a,:] = T[a].reshape(-1)
                
                # Build design matrix for U update: shape (mp*np_, r)
                # Column r is V[:,r] * W[:,r] (element-wise product, vectorized)
                D_U = np.zeros((mp * np_, r))
                for r_idx in range(r):
                    D_U[:, r_idx] = (V[:, r_idx] * W[:, r_idx]).flatten()
                
                # Solve for each row of U
                for a in range(nm):
                    rhs = T[a, :, :].flatten()
                    try:
                        U[a, :] = np.linalg.lstsq(D_U, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update V: for each row b, solve least squares
                # T[:,b,:] = sum_r V[b,r] * outer(U[:,r], W[:,r])
                D_V = np.zeros((nm * np_, r))
                for r_idx in range(r):
                    D_V[:, r_idx] = (U[:, r_idx] * W[:, r_idx]).flatten()
                
                for b in range(mp):
                    rhs = T[:, b, :].flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(D_V, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update W: for each row c, solve least squares
                # T[:,:,c] = sum_r W[c,r] * outer(U[:,r], V[:,r])
                D_W = np.zeros((nm * mp, r))
                for r_idx in range(r):
                    D_W[:, r_idx] = (U[:, r_idx] * V[:, r_idx]).flatten()
                
                for c in range(np_):
                    rhs = T[:, :, c].flatten()
                    try:
                        W[c, :] = np.linalg.lstsq(D_W, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(recon - T)
                
                if err < 1e-10:
                    break
                
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-8:
                break
        
        if best_err > 1e-6:
            return None
        
        # Try to round to integers or half-integers
        U_r, V_r, W_r = best_solution
        
        # Try different scales
        for scale in [0.5, 1.0, 2.0, 0.25, 4.0, 0.125, 8.0]:
            U_scaled = U_r / scale
            V_scaled = V_r / scale
            W_scaled = W_r / scale
            
            # Round to nearest half-integer
            U_rounded = np.round(U_scaled * 2) / 2
            V_rounded = np.round(V_scaled * 2) / 2
            W_rounded = np.round(W_scaled * 2) / 2
            
            # Check if rounded solution is exact
            recon_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            err_rounded = np.linalg.norm(recon_rounded - T)
            
            if err_rounded < 1e-8:
                # Convert to lists
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
        
        return None
    
    # Try rank 25
    result = try_rank(25)
    if result is not None:
        return result
    
    # Try rank 26 as fallback
    result = try_rank(26)
    if result is not None:
        return result
    
    # Fall back to schoolbook
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
