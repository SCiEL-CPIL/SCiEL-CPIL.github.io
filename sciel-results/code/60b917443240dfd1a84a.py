import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build target tensor
    T_target = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reconstruct
    T_recon = np.zeros((nm, mp, np_))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    if np.linalg.norm(T_recon - T_target) > tol:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Build target tensor
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook rank
    rank_sb = n * m * p
    
    # Prepare schoolbook fallback
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((np_, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    def snap_to_half(x):
        return np.round(x * 2) / 2.0

    def compute_error(U, V, W, rank):
        """Compute the error of the decomposition."""
        T_recon = np.zeros((nm, mp, np_))
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.linalg.norm(T - T_recon)

    def try_rank(r, time_budget, init_type="random"):
        """Try to find a rank-r decomposition."""
        best_err = np.inf
        best_result = None
        
        max_restarts = 10 if init_type == "structured" else 20
        max_iters = 300
        
        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            
            # Initialize
            if init_type == "structured":
                # Structured initialization based on tensor pattern
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((np_, r))
                
                # Assign columns to cover the tensor structure
                col_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col_idx < r:
                                U[i*m+j, col_idx] = 1
                                V[j*p+k, col_idx] = 1
                                W[k*n+i, col_idx] = 1
                                col_idx += 1
                
                # Add random perturbations to escape local minima
                U += np.random.randn(nm, r) * 0.1
                V += np.random.randn(mp, r) * 0.1
                W += np.random.randn(np_, r) * 0.1
            else:
                # Random initialization
                U = np.random.randn(nm, r) * 0.3
                V = np.random.randn(mp, r) * 0.3
                W = np.random.randn(np_, r) * 0.3
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > time_budget:
                    break
                
                # Compute current error
                err = compute_error(U, V, W, r)
                
                if err < 1e-10:
                    return U.copy(), V.copy(), W.copy()
                
                if err < best_err:
                    best_err = err
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if err > prev_err * (1 + 1e-6):
                    break
                prev_err = err
                
                # Update one column at a time using SVD
                idx = it % r
                # Compute residual without this column
                T_other = T.copy()
                for rr in range(r):
                    if rr == idx:
                        continue
                    T_other -= np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                
                # SVD decomposition of the residual
                Tf = T_other.reshape(nm, mp*np_)
                try:
                    U_, S_, Vt_ = np.linalg.svd(Tf, full_matrices=False)
                    if S_[0] < 1e-15:
                        continue
                    u_vec = U_[:, 0] * np.sqrt(S_[0])
                    v_vec_full = Vt_[0, :] * np.sqrt(S_[0])
                    V_mat = v_vec_full.reshape(mp, np_)
                    V2_, S2_, Wt_ = np.linalg.svd(V_mat, full_matrices=False)
                    if S2_[0] < 1e-15:
                        continue
                    v2_vec = V2_[:, 0] * np.sqrt(S2_[0])
                    w_vec = Wt_[0, :] * np.sqrt(S2_[0])
                    
                    U[:, idx] = u_vec
                    V[:, idx] = v2_vec
                    W[:, idx] = w_vec
                except Exception:
                    continue
        
        if best_result is not None and best_err < 1e-9:
            return best_result
        return None

    # Try ranks from lower to higher
    # Start from a reasonable lower bound
    min_rank = max(nm, mp, np_)  # 20
    
    best_result = None
    for r in range(min_rank, rank_sb + 1):
        if time.time() > deadline - 20:
            break
        
        time_budget = min(deadline - 5, time.time() + 20)
        
        # First try structured initialization
        result = try_rank(r, time_budget, init_type="structured")
        
        if result is None and time.time() < deadline - 10:
            # Then try random initialization
            time_budget2 = min(deadline - 5, time.time() + 15)
            result = try_rank(r, time_budget2, init_type="random")
        
        if result is not None:
            U, V, W = result
            U_s = snap_to_half(U)
            V_s = snap_to_half(V)
            W_s = snap_to_half(W)
            
            if _check_decomposition(U_s, V_s, W_s, n, m, p):
                best_result = (U_s, V_s, W_s)
                break
            elif _check_decomposition(U, V, W, n, m, p):
                best_result = (U, V, W)
                break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
