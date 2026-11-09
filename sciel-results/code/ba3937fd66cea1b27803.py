"""Program for matrix multiplication tensor decomposition of (3,3,8)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 15.0


def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal entries are zero
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    for kn in range(p):
                        for in_ in range(n):
                            if jp == j and kp == kn and in_ == i:
                                continue  # This is a diagonal entry, already checked
                            val = np.sum(U_arr[i*m+j, :] * V_arr[jp*p+kp, :] * W_arr[kn*n+in_, :])
                            if abs(val) > tol:
                                return False
    
    return True


def _greedy_decomposition(n, m, p, max_rank=None, deadline=None):
    """Greedy decomposition: iteratively find best rank-1 component for residual.
    
    Returns (U, V, W) as lists of lists, or None if it fails.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    T = _build_target_tensor(n, m, p)
    residual = T.copy()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    start = time.time()
    
    for step in range(max_rank):
        # Check deadline
        if deadline is not None and time.time() > deadline:
            if not U_cols:
                return None
            break
        
        # Find best rank-1 approximation of residual
        # Use ALS: initialize randomly, then alternate optimizing u, v, w
        best_u, best_v, best_w = None, None, None
        best_score = -1.0
        
        # Try multiple random initializations
        n_init = 20
        for init in range(n_init):
            if deadline is not None and time.time() > deadline:
                break
                
            np.random.seed(step * 1000 + init)
            u = np.random.randn(n * m)
            v = np.random.randn(m * p)
            w = np.random.randn(p * n)
            
            # ALS iterations
            for als_iter in range(30):
                # Optimize u: u = argmax_u <residual, u ⊗ v ⊗ w> / ||v|| ||w||
                grad_u = np.einsum('abc,b,c->a', residual, v, w)
                norm_vw = np.linalg.norm(v) * np.linalg.norm(w)
                if norm_vw > 1e-15:
                    u = grad_u / norm_vw
                
                # Optimize v
                grad_v = np.einsum('abc,a,c->b', residual, u, w)
                norm_uw = np.linalg.norm(u) * np.linalg.norm(w)
                if norm_uw > 1e-15:
                    v = grad_v / norm_uw
                
                # Optimize w
                grad_w = np.einsum('abc,a,b->c', residual, u, v)
                norm_uv = np.linalg.norm(u) * np.linalg.norm(v)
                if norm_uv > 1e-15:
                    w = grad_w / norm_uv
            
            # Score: how well u⊗v⊗w explains residual
            score = np.einsum('abc,a,b,c', residual, u, v, w)
            
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_u is None or best_score <= 1e-12:
            break
        
        # Scale: alpha = <res, u⊗v⊗w> / ||u⊗v⊗w||^2
        u_norm_sq = np.dot(best_u, best_u)
        v_norm_sq = np.dot(best_v, best_v)
        w_norm_sq = np.dot(best_w, best_w)
        
        if u_norm_sq * v_norm_sq * w_norm_sq < 1e-30:
            break
        
        alpha = best_score / (u_norm_sq * v_norm_sq * w_norm_sq)
        
        # Subtract from residual
        residual -= alpha * np.einsum('a,b,c->abc', best_u, best_v, best_w)
        
        # Store the component
        U_cols.append(best_u.tolist())
        V_cols.append(best_v.tolist())
        W_cols.append((best_w * alpha).tolist())
        
        # Check if residual is negligible
        res_norm = np.linalg.norm(residual)
        if res_norm < 1e-10:
            break
        
        # Check deadline
        if deadline is not None and time.time() > deadline:
            break
    
    if not U_cols:
        return None
    
    # Build matrices
    rank = len(U_cols)
    U = [[U_cols[r][a] for r in range(rank)] for a in range(n * m)]
    V = [[V_cols[r][b] for r in range(rank)] for b in range(m * p)]
    W = [[W_cols[r][c] for r in range(rank)] for c in range(p * n)]
    
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as a safe fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    # Try greedy decomposition with various target ranks
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    # Try ranks from low to high, starting with a reasonable lower bound
    for target_rank in range(40, min(72, best_rank)):
        if time.time() > deadline:
            break
        
        try:
            result = _greedy_decomposition(n, m, p, max_rank=target_rank, deadline=deadline)
            if result is not None:
                U_g, V_g, W_g = result
                if _verify_decomposition(U_g, V_g, W_g, n, m, p):
                    rank_g = len(W_g[0]) if W_g else 0
                    if rank_g < best_rank:
                        best_result = {"U": U_g, "V": V_g, "W": W_g}
                        best_rank = rank_g
                        break
        except Exception:
            pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - SAFETY_MARGIN:
        return best_result
    
    return best_result
