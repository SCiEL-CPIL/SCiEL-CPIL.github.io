import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Uses alternating projection with auxiliary variables to search for rank-20 decomposition.
    Falls back to Strassen's algorithm (rank 21) if no rank-20 solution is found.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Flatten tensor for linear algebra
    T_flat = T.flatten()
    
    # Precompute the non-zero pattern
    nz_indices = np.argwhere(T != 0)
    nz_count = len(nz_indices)  # Should be 24
    
    def flatten_UVW(U, V, W, rank):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten_UVW(x, rank):
        size_U = nm * rank
        size_V = mp * rank
        U = x[:size_U].reshape(nm, rank)
        V = x[size_U:size_U+size_V].reshape(mp, rank)
        W = x[size_U+size_V:].reshape(pn, rank)
        return U, V, W
    
    def compute_residual(U, V, W):
        """Compute T_approx - T using einsum"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_approx - T).flatten()
    
    def alternating_projection(U, V, W, rank, max_iters=200, tol=1e-10):
        """
        Alternating least squares: fix two factors, solve for the third.
        The problem is linear in each factor given the other two.
        """
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Fix V, W; solve for U
            # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed V,W: T_approx[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
            # This is linear in U. We can solve via least squares.
            # Build matrix: for each (a,b,c), coefficient of U[a,r] is V[b,r]*W[c,r]
            # Reshape: for each a, we have equations over (b,c,r)
            
            # More efficient: use the fact that for each a, 
            # sum_r U[a,r] * V[b,r] * W[c,r] should equal T[a,b,c]
            # Let M_a[r, (b,c)] = V[b,r] * W[c,r], shape (rank, mp*pn)
            # Then U[a] @ M_a = T[a].flatten()
            # Solve: U[a] = lstsq(M_a.T, T[a].flatten())
            
            new_U = np.zeros_like(U)
            for a in range(nm):
                # M_a has shape (rank, mp*pn) where M_a[r, b*pn+c] = V[b,r]*W[c,r]
                M_a = (V.T @ W.T).T  # shape (rank, mp, pn) -> need to reshape
                # Actually: M_a[r, (b,c)] = V[b,r] * W[c,r]
                M_a = np.einsum('br,cr->rbc', V, W).reshape(rank, mp * pn)
                rhs = T[a].flatten()
                # Solve M_a.T @ U[a] = rhs, i.e., (mp*pn x rank) @ (rank x 1) = (mp*pn x 1)
                # Use lstsq
                try:
                    sol, _, _, _ = np.linalg.lstsq(M_a.T, rhs, rcond=None)
                    new_U[a] = sol
                except:
                    new_U[a] = U[a]
            U = new_U
            
            # Fix U, W; solve for V
            new_V = np.zeros_like(V)
            for b in range(mp):
                # M_b[r, (a,c)] = U[a,r] * W[c,r]
                M_b = np.einsum('ar,cr->rac', U, W).reshape(rank, nm * pn)
                rhs = T[:, b, :].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M_b.T, rhs, rcond=None)
                    new_V[b] = sol
                except:
                    new_V[b] = V[b]
            V = new_V
            
            # Fix U, V; solve for W
            new_W = np.zeros_like(W)
            for c in range(pn):
                # M_c[r, (a,b)] = U[a,r] * V[b,r]
                M_c = np.einsum('ar,br->rab', U, V).reshape(rank, nm * mp)
                rhs = T[:, :, c].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M_c.T, rhs, rcond=None)
                    new_W[c] = sol
                except:
                    new_W[c] = W[c]
            W = new_W
            
            # Check convergence
            if (iteration > 10):
                err = np.max(np.abs(compute_residual(U, V, W)))
                if err < tol:
                    break
        
        return U, V, W
    
    def polish(U, V, W, rank, max_iters=100):
        """High-precision polish using alternating projection with tighter tolerance"""
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Fix V, W; solve for U with high precision
            new_U = np.zeros_like(U)
            for a in range(nm):
                M_a = np.einsum('br,cr->rbc', V, W).reshape(rank, mp * pn)
                rhs = T[a].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M_a.T, rhs, rcond=1e-15)
                    new_U[a] = sol
                except:
                    new_U[a] = U[a]
            U = new_U
            
            # Fix U, W; solve for V
            new_V = np.zeros_like(V)
            for b in range(mp):
                M_b = np.einsum('ar,cr->rac', U, W).reshape(rank, nm * pn)
                rhs = T[:, b, :].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M_b.T, rhs, rcond=1e-15)
                    new_V[b] = sol
                except:
                    new_V[b] = V[b]
            V = new_V
            
            # Fix U, V; solve for W
            new_W = np.zeros_like(W)
            for c in range(pn):
                M_c = np.einsum('ar,br->rab', U, V).reshape(rank, nm * mp)
                rhs = T[:, :, c].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M_c.T, rhs, rcond=1e-15)
                    new_W[c] = sol
                except:
                    new_W[c] = W[c]
            W = new_W
        
        return U, V, W
    
    # Attempt to find rank 20 decomposition via alternating projection
    rank_try = 20
    best_result = None
    best_error = np.inf
    
    num_attempts = 15
    for attempt in range(num_attempts):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(12345 + attempt * 7919)
        scale = 0.5 + 0.1 * attempt
        U0 = np.random.randn(nm, rank_try) * scale
        V0 = np.random.randn(mp, rank_try) * scale
        W0 = np.random.randn(pn, rank_try) * scale
        
        try:
            U, V, W = alternating_projection(U0, V0, W0, rank_try, max_iters=150, tol=1e-10)
            err = np.max(np.abs(compute_residual(U, V, W)))
            
            if err < best_error:
                best_error = err
                best_result = (U.copy(), V.copy(), W.copy())
                
                # If we got close, do a polish pass
                if err < 1e-4:
                    U_p, V_p, W_p = polish(U, V, W, rank_try, max_iters=50)
                    err_p = np.max(np.abs(compute_residual(U_p, V_p, W_p)))
                    if err_p < best_error:
                        best_error = err_p
                        best_result = (U_p.copy(), V_p.copy(), W_p.copy())
                        
        except Exception:
            continue
    
    # Check if we found a valid rank-20 solution
    if best_result is not None and best_error < 1e-8:
        U, V, W = best_result
        T_check = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_check, T, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Strassen rank 21
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    def map_a(local_idx):
        return local_idx
        
    def map_b(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return 0 * 6 + k0
        elif local_idx == 1:
            return 0 * 6 + k1
        elif local_idx == 2:
            return 1 * 6 + k0
        else:
            return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return k0 * 2 + 0
        elif local_idx == 1:
            return k1 * 2 + 0
        elif local_idx == 2:
            return k0 * 2 + 1
        else:
            return k1 * 2 + 1

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    U[map_a(local_idx), r] = u_val
                    
                v_val = step['v'][local_idx]
                if v_val != 0:
                    V[map_b(local_idx, g), r] = v_val
                    
                w_val = step['w'][local_idx]
                if w_val != 0:
                    W[map_c(local_idx, g), r] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
