"""Matrix multiplication 3,3,4 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _build_target_tensor(n=3, m=3, p=4):
    """Build the multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n=3, m=3, p=4, tol=1e-9):
    """Verify that sum_c U[a,c]*V[b,c]*W[c_idx,c] equals the multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    if U_arr.shape != (n*m, U_arr.shape[1]) or V_arr.shape != (m*p, U_arr.shape[1]) or W_arr.shape != (p*n, U_arr.shape[1]):
        return False
    T_prime = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    T = _build_target_tensor(n, m, p)
    return np.allclose(T_prime, T, atol=tol)

def _schoolbook_decomposition(n=3, m=3, p=4):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def _als_decomposition(n, m, p, rank, max_iter=2000, tol=1e-10, seed=42):
    """
    Alternating Least Squares for tensor rank decomposition.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rng = np.random.RandomState(seed)
    T = _build_target_tensor(n, m, p)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    prev_err = np.inf
    for it in range(max_iter):
        # Update U: for each a, solve V^T diag(W^T T[a,:,:]) V u_a = V^T (W^T T[a,:,:]) vec
        # More efficiently: for each r, the residual for U[:,r] given V and W
        # T[a,b,c] - sum_{r' != r} U[a,r']*V[b,r']*W[c,r'] = U[a,r] * V[b,r] * W[c,r]
        # So U[a,r] = <T - residual_others, V[:,r] W[:,r]> / <V[:,r] W[:,r], V[:,r] W[:,r]>
        
        # Efficient approach:
        # For updating U: 
        #   For each r: U[:,r] = argmin sum_{a,b,c} (T[a,b,c] - sum_r' U[a,r']V[b,r']W[c,r'])^2
        #   = argmin_a sum_{b,c} (T[a,b,c] - sum_{r' != r} U[a,r']V[b,r']W[c,r'])^2 * (V[b,r]W[c,r])^2
        #   This is a linear system: U[:,r] = (V_r W_r)^T T_a / ||V_r W_r||^2 where T_a is adjusted
        
        # Let's use the standard ALS formulation:
        # Fix V, W. For each a:
        #   sum_r U[a,r] * (V[b,r] W[c,r]) = T[a,b,c] for all b,c
        #   => sum_r U[a,r] * (V[b,r] W[c,r]) = T[a,b,c]
        #   Let M[a] be the matrix with M[a][(b,c), r] = V[b,r] * W[c,r]
        #   Then M[a] @ U[a,:] = T[a,:,:].ravel()
        #   U[a,:] = lstsq(M[a], T[a,:,:].ravel())
        
        # Similarly for V and W.
        
        # Precompute M_U[a] = kron(W, V) reshaped: M_U[a][(b*p+c), r] = V[b,r]*W[c,r]
        # Actually M_U doesn't depend on a, it's the same for all a!
        # M_U[(b,c), r] = V[b,r] * W[c,r]
        
        M_U = np.einsum('br,cr->bcr', V, W).reshape(m*p, rank)
        # For each a: U[a,:] solves M_U @ U[a,:] = T[a,:,:].ravel()
        T_flat = T.reshape(n*m, m*p*p*n)
        # U[a,:] = lstsq(M_U, T_flat[a])
        U_new = np.zeros((n*m, rank))
        for a in range(n*m):
            U_new[a] = np.linalg.lstsq(M_U, T_flat[a], rcond=None)[0]
        
        # Update V: M_V[(a*p+c), r] = U[a,r] * W[c,r]
        M_V = np.einsum('ar,cr->acr', U_new, W).reshape(n*m*p, rank)
        # T reshaped: T[a, b, c] -> for fixed b: T[:, b, :].ravel() has shape (n*m*p,)
        # M_V @ V[b,:] = T[:, b, :].ravel()
        V_new = np.zeros((m*p, rank))
        for b in range(m*p):
            # T[:, b, :] has shape (n*m, p), ravel to (n*m*p,)
            rhs = T[:, b, :].ravel()
            V_new[b] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
        
        # Update W: M_W[(a*m+b), r] = U[a,r] * V[b,r]
        M_W = np.einsum('ar,br->abr', U_new, V_new).reshape(n*m*m*p, rank)
        # T[a, b, c] -> for fixed c: T[:, :, c].ravel() has shape (n*m*m*p,)
        W_new = np.zeros((p*n, rank))
        for c in range(p*n):
            rhs = T[:, :, c].ravel()
            W_new[c] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
        
        # Compute error
        T_prime = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
        err = np.linalg.norm(T_prime - T)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
        U, V, W = U_new, V_new, W_new
    
    # Final verification
    T_prime = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_prime, T, atol=1e-8):
        return None
    
    # Round to nearest half-integer if close
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    T_prime_r = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
    if np.allclose(T_prime_r, T, atol=1e-8):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Try rounding to integers
    U_rounded = np.round(U)
    V_rounded = np.round(V)
    W_rounded = np.round(W)
    T_prime_r = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
    if np.allclose(T_prime_r, T, atol=1e-8):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    return None

def _random_search(n, m, p, target_rank, max_iter=5000, seed=42):
    """Try multiple random initializations with ALS."""
    best = None
    for trial in range(max_iter // 100 + 1):
        s = seed + trial * 1000
        result = _als_decomposition(n, m, p, target_rank, max_iter=max_iter, seed=s)
        if result is not None:
            return result
    return None

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    # Start with schoolbook
    best_result = _schoolbook_decomposition()
    best_rank = 36
    
    n, m, p = 3, 3, 4
    
    # Try ALS for decreasing ranks
    for target_rank in range(35, 20, -1):
        if time.time() > deadline - 30:
            break
        result = _random_search(n, m, p, target_rank, max_iter=2000, seed=42)
        if result is not None:
            # Verify
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                best_result = result
                best_rank = target_rank
                break
    
    return best_result
