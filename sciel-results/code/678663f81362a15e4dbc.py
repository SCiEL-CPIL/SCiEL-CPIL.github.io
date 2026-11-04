"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[a,b,c]."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    T = _build_tensor(n, m, p)
    # Compute sum_r U[:,r] outer V[:,r] outer W[:,r]
    # This is a 3-tensor
    # Efficient: for each r, add outer product
    T_recon = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(T_recon, T, atol=tol, rtol=0)

def _als_step(U, V, W, T, n, m, p):
    """One alternating least squares step. Returns new U, V, W."""
    rank = U.shape[1]
    # Fix U, V: solve for W
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For each c: sum_r (sum_a T[a,b,c] * ... ) -- this is not quite linear in W alone
    # Actually: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For fixed U,V: this is linear in W. 
    # T[a,b,c] = sum_r W[c,r] * (U[a,r]*V[b,r])
    # So for each c, we have a vector equation: T[:, :, c] = sum_r W[c,r] * (U[:,r] * V[:,r])
    # This is an overdetermined linear system for W[c,:].
    
    # Build the linear system for W
    # For each c, we need: sum_r W[c,r] * (U[:,r] * V[:,r]) = T[:,:,c]
    # This is: M_c @ W[c,:] = T[:,:,c] where M_c[r, (a,b)] = U[a,r]*V[b,r]
    # But T[:,:,c] is a 2D array of size (n*m, m*p), and we're matching it to a sum of outer products.
    
    # Actually, let me flatten: for each c, T[:,:,c] is a matrix of size (n*m) x (m*p).
    # We want: T[:,:,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
    # This is a matrix equation. Flatten both sides:
    # vec(T[:,:,c]) = sum_r W[c,r] * vec(U[:,r] outer V[:,r])
    # So: vec(T[:,:,c]) = A_c @ W[c,:] where A_c[:, r] = vec(U[:,r] outer V[:,r])
    
    # But this is a large system. Let me use a different approach.
    # Actually, the standard ALS for tensor decomposition:
    # Fix V, W: solve for U. T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
    # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
    # Flatten: vec(T[a,:,:]) = sum_r U[a,r] * vec(V[:,r] outer W[:,r])
    # So: vec(T[a,:,:]) = B_a @ U[a,:] where B_a[:,r] = vec(V[:,r] outer W[:,r])
    
    # Let me just do it with np.linalg.lstsq for each row.
    
    # Solve for W: fix U, V
    W_new = np.zeros_like(W)
    for c in range(p*n):
        # T[:,:,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
        # Flatten: T[:,:,c].ravel() = sum_r W[c,r] * (U[:,r] * V[:,r]).ravel()  -- NO
        # U[:,r] outer V[:,r] is a (n*m) x (m*p) matrix. Its (a,b) entry is U[a,r]*V[b,r].
        # So T[:,:,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
        # Flatten: T[:,:,c].ravel() = sum_r W[c,r] * (U[:,r] outer V[:,r]).ravel()
        # The vector (U[:,r] outer V[:,r]).ravel() has entries U[a,r]*V[b,r] for all (a,b).
        # So we're solving: for each (a,b), T[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
        # This is: T[:,:,c] = sum_r (U[:,r] * W[c,r]) outer V[:,r] ... no.
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed U,V: T[a,b,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
        # So for each c, we have a matrix equation: T[:,:,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
        # Flatten: T[:,:,c].ravel() = sum_r W[c,r] * (U[:,r] outer V[:,r]).ravel()
        # Let M_c be a matrix with columns M_c[:,r] = (U[:,r] outer V[:,r]).ravel()
        # Then T[:,:,c].ravel() = M_c @ W[c,:]
        
        M_c = np.zeros((n*m*m*p, rank))
        for r in range(rank):
            M_c[:, r] = (U[:, r] * V[:, r]).ravel()  # Wait, this is element-wise product, not outer
            # Actually (U[:,r] outer V[:,r]).ravel() = np.outer(U[:,r], V[:,r]).ravel()
            M_c[:, r] = np.outer(U[:, r], V[:, r]).ravel()
        
        rhs = T[:, :, c].ravel()
        sol, residuals, rank_c, sv = np.linalg.lstsq(M_c, rhs, rcond=None)
        W_new[c, :] = sol
    
    # Solve for U: fix V, W
    U_new = np.zeros_like(U)
    for a in range(n*m):
        # T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
        M_a = np.zeros((m*p*p*n, rank))
        for r in range(rank):
            M_a[:, r] = np.outer(V[:, r], W[:, r]).ravel()
        rhs = T[a, :, :].ravel()
        sol, residuals, rank_a, sv = np.linalg.lstsq(M_a, rhs, rcond=None)
        U_new[a, :] = sol
    
    # Solve for V: fix U, W
    V_new = np.zeros_like(V)
    for b in range(m*p):
        # T[:,b,:] = sum_r V[b,r] * (U[:,r] outer W[:,r])
        M_b = np.zeros((n*m*p*n, rank))
        for r in range(rank):
            M_b[:, r] = np.outer(U[:, r], W[:, r]).ravel()
        rhs = T[:, b, :].ravel()
        sol, residuals, rank_b, sv = np.linalg.lstsq(M_b, rhs, rcond=None)
        V_new[b, :] = sol
    
    return U_new, V_new, W_new

def _search_rank(n, m, p, target_rank, max_iters=200, n_restarts=10, time_budget=120.0):
    """Search for a decomposition of the given rank using ALS."""
    T = _build_tensor(n, m, p)
    start_time = time.time()
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget - 10:
            break
        
        # Random initialization
        rng = np.random.RandomState(42 + restart)
        U = rng.randn(n*m, target_rank)
        V = rng.randn(m*p, target_rank)
        W = rng.randn(p*n, target_rank)
        # Scale to make initial residual reasonable
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        U *= (n*m)**0.5
        V *= (m*p)**0.5
        W *= (p*n)**0.5
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget - 10:
                break
            
            U, V, W = _als_step(U, V, W, T, n, m, p)
            
            # Compute residual
            T_recon = np.zeros_like(T)
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err = np.linalg.norm(T_recon - T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def _to_list(mat):
    """Convert numpy array to list of lists with integer/half-integer entries if possible."""
    mat = np.array(mat)
    # Try to round to nearest half-integer
    rounded = np.round(mat * 2) / 2
    if np.allclose(mat, rounded, atol=1e-6):
        mat = rounded
    return mat.tolist()

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from lower to higher
    # Known: 3x3x3x3 has rank 23. For 3x3x3x6, try ranks below 54.
    # Let's try a few candidate ranks.
    candidate_ranks = [46, 48, 50, 52, 53]
    
    for rank in candidate_ranks:
        try:
            U, V, W, err = _search_rank(n, m, p, rank, max_iters=100, n_restarts=5, time_budget=100.0)
            if U is not None and err < 1e-6:
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
        except Exception:
            continue
    
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
