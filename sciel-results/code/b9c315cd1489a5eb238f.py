import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    D = np.zeros((n, m, p))
    for c in range(r):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
    return np.max(np.abs(D - 1.0)) < tol

def _build_direct_sum(U1, V1, W1, U2, V2, W2, n1, n2, m, p1, p2, n, p):
    """
    Build direct sum decomposition for (n1+n2)xm x (p1+p2) from two sub-decompositions.
    
    Sub1: n1 x m x p1 with factors U1, V1, W1
    Sub2: n2 x m x p2 with factors U2, V2, W2
    
    The combined tensor is block-diagonal in the (i,k) sense:
    T[i,j,k] = 1 if (i < n1 and k < p1) or (i >= n1 and k >= p1)
    But actually for full 3x4x6 we need ALL entries = 1.
    """
    # This direct sum approach only works for block-diagonal tensors, not the full all-ones tensor.
    # So this function is actually not applicable for the full 3x4x6 problem.
    # Returning None to indicate this approach doesn't work for the full tensor.
    return None

def construct():
    n, m, p = 3, 4, 6
    start_time = time.time()
    deadline = start_time + 130.0
    
    # The schoolbook rank is n*m*p = 72
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition as fallback
    def _schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try to find lower rank decompositions using gradient descent
    # The key insight: we can use random restarts and local optimization
    
    np.random.seed(42)
    
    def _residuals(params, r, n, m, p):
        """Compute residuals for rank-r decomposition."""
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[0:size_u*r].reshape(size_u, r)
        V = params[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
        W = params[size_u*r + size_v*r:].reshape(size_w, r)
        
        D = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
        
        return (D - 1.0).flatten()
    
    def _try_rank(r, n, m, p, max_restarts=3, max_iter=2000, tol=1e-6):
        """Try to find a rank-r decomposition."""
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_params = r * (size_u + size_v + size_w)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            if restart == 0:
                # Try to initialize closer to a valid solution
                params0 = np.zeros(total_params)
                # Set up some initial structure
                for c in range(min(r, size_u)):
                    params0[c * size_u + c] = 1.0
            else:
                params0 = np.random.randn(total_params) * 0.5
            
            # Use scipy's minimize
            try:
                from scipy.optimize import minimize
                
                def objective(params):
                    res = _residuals(params, r, n, m, p)
                    return np.sum(res**2)
                
                result = minimize(objective, params0, method='Nelder-Mead', 
                                  options={'maxiter': max_iter, 'fatol': tol, 'xatol': tol})
                
                if result.fun < tol:
                    U = result.x[0:size_u*r].reshape(size_u, r)
                    V = result.x[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
                    W = result.x[size_u*r + size_v*r:].reshape(size_w, r)
                    
                    # Verify
                    if _verify_decomposition(U, V, W, n, m, p, tol=1e-5):
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception:
                continue
        
        return None
    
    # Try ranks from 71 down to some minimum
    # For 3x4x6, the minimum rank is at most 72 (schoolbook)
    # Known results suggest it might be lower but we don't know exactly
    
    # Given time constraints, try a few promising ranks
    ranks_to_try = [71, 70, 69, 68, 67, 66]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        result = _try_rank(r, n, m, p, max_restarts=2, max_iter=1000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If no lower rank found, return schoolbook
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
