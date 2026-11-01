import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook decomposition (rank 30)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        """Check if U, V, W correctly decompose T"""
        rank = U.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        return np.allclose(T_recon, T, atol=1e-8)
    
    def round_to_half(x):
        return np.round(x * 2) / 2.0
    
    # The known optimal rank for 2x3x5 is 28.
    # We attempt to find this decomposition using an iterative refinement approach.
    # Since the problem is non-convex and coupled, we use a heuristic:
    # Start with a rank-30 solution and try to "merge" pairs of terms to reduce rank.
    # This is equivalent to finding a rank-28 solution by identifying linear dependencies.
    
    # Alternative: Use a global optimization approach with multiple restarts.
    # Given the time constraint, we'll try a few random initializations for rank 28.
    
    def try_rank_28():
        rank = 28
        nm, mp, pn = n*m, m*p, p*n
        n_vars = rank * (nm + mp + pn)
        
        # Precompute T_flat for faster residual computation
        T_flat = T.flatten()
        
        def residuals(x):
            U = x[:rank*nm].reshape(nm, rank)
            V = x[rank*nm:rank*(nm+mp)].reshape(mp, rank)
            W = x[rank*(nm+mp):].reshape(pn, rank)
            
            # Efficient tensor reconstruction
            # T_recon[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # We can compute this as:
            # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_recon - T).flatten()
        
        best_err = np.inf
        best_x = None
        
        # Try several random initializations
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            
            # Initialize near the schoolbook solution but with rank 28
            # We can't directly initialize from rank 30, so use random init
            # with small values to help convergence
            x0 = np.random.randn(n_vars) * 0.1
            
            # Use scipy's least_squares if available
            try:
                from scipy.optimize import least_squares
                result = least_squares(residuals, x0, method='lm', max_nfev=2000)
                err = np.max(np.abs(result.fun))
                
                if err < best_err:
                    best_err = err
                    best_x = result.x.copy()
                    
                    if err < 1e-6:
                        # Found a good solution, round to half-integers
                        U = best_x[:rank*nm].reshape(nm, rank)
                        V = best_x[rank*nm:rank*(nm+mp)].reshape(mp, rank)
                        W = best_x[rank*(nm+mp):].reshape(pn, rank)
                        
                        U_r = round_to_half(U)
                        V_r = round_to_half(V)
                        W_r = round_to_half(W)
                        
                        if verify(U_r, V_r, W_r):
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
            except Exception:
                pass
        
        # If we found a solution with low error but rounding failed, try without rounding
        if best_x is not None and best_err < 1e-4:
            U = best_x[:rank*nm].reshape(nm, rank)
            V = best_x[rank*nm:rank*(nm+mp)].reshape(mp, rank)
            W = best_x[rank*(nm+mp):].reshape(pn, rank)
            
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try to find rank 28
    if time.time() < deadline - 15:
        result_28 = try_rank_28()
        if result_28 is not None:
            return result_28
    
    # Fallback to schoolbook
    return best_result
