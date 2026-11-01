import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Target tensor: T[a,b,c] where a = i*m+j (i in 0..n-1, j in 0..m-1)
    # b = j*p+k (j in 0..m-1, k in 0..p-1)  
    # c = k*n+i (k in 0..p-1, i in 0..n-1)
    # T[a,b,c] = 1 if the indices are consistent (same j and i) else 0
    
    # Actually, the multiplication tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor entry for (i,j,k) is 1 (it's the coefficient)
    # In the factorization: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for valid (i,j,k), 0 otherwise
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def try_rank(r, max_iters=500, tol=1e-6):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline - 2:
            return None
        
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Update U: for each a, U[a,:] should satisfy sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
            # Fix V,W, solve for U[a,:]
            # For fixed a: sum_t U[a,t] * (sum_{b,c} T[a,b,c] * V[b,t]*W[c,t] / something)
            # Actually: for each a, we need: sum_t U[a,t] * X[a,t] = ... no
            # The equation is: for all (a,b,c): sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
            # For fixed a, varying (b,c): this is a linear system in U[a,:]
            # Let M_a be a matrix of size (m*p*p*n) x r where M_a[(b,c),t] = V[b,t]*W[c,t]
            # Then M_a @ U[a,:] should equal T[a,:,:].ravel()
            
            # This is expensive. Let's use a simpler ALS:
            # Update U: for each a, solve least squares
            for a in range(n*m):
                # Build system: for each (b,c), sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
                # This is: sum_t U[a,t] * (V[b,t]*W[c,t]) = T[a,b,c]
                # Let vec be over (b,c): the coefficient for U[a,t] is V[b,t]*W[c,t]
                # So we have a system: A @ U[a,:] = rhs where A has shape (m*p*p*n, r)
                A = np.zeros((m*p*p*n, r))
                rhs = T[a].ravel()
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A[idx, :] = V[b, :] * W[c, :]
                        idx += 1
                # Solve least squares
                U[a, :], residuals, rank_a, sv = np.linalg.lstsq(A, rhs, rcond=None)
            
            # Update V
            for b in range(m*p):
                A = np.zeros((n*m*p*n, r))
                rhs = T[:, b, :].reshape(-1)  # shape (n*m*p*n,)
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                V[b, :], residuals, rank_b, sv = np.linalg.lstsq(A, rhs, rcond=None)
            
            # Update W
            for c in range(p*n):
                A = np.zeros((n*m*m*p, r))
                rhs = T[:, :, c].reshape(-1)  # shape (n*m*m*p,)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                W[c, :], residuals, rank_c, sv = np.linalg.lstsq(A, rhs, rcond=None)
            
            # Check convergence
            err = np.abs(U @ np.einsum('bt,ct->bc', V, W) - T).max() if False else 0
            # Compute actual error
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.abs(recon - T).max()
            if err < tol:
                break
        
        # Check final error
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.abs(recon - T).max()
        if err > 1e-4:
            return None
        
        # Try to round to integer/half-integer
        # Scale to find common denominator
        all_vals = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        # Try rounding to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        recon_r = np.einsum('at,bt,ct->abc', U_r, V_r, W_r)
        if np.abs(recon_r - T).max() < 1e-6:
            return (U_r, V_r, W_r)
        
        # Try rounding to nearest integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        recon_r = np.einsum('at,bt,ct->abc', U_r, V_r, W_r)
        if np.abs(recon_r - T).max() < 1e-6:
            return (U_r, V_r, W_r)
        
        # Try to find a rational approximation
        # For now, if error is small but not exact, try harder
        return None
    
    # Try ranks from low to high
    best_result = None
    for r in range(20, 37):
        if time.time() > deadline - 5:
            break
        result = try_rank(r)
        if result is not None:
            best_result = (r, result)
            break
    
    if best_result is None:
        # Fallback to schoolbook
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (r, (U, V, W))
    
    r, (U, V, W) = best_result
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
