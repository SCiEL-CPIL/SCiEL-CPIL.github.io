"""Combined: rank-18 base with robust LM-based rank-16/17 search."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=1e-9)

def _build_rank18(n, m, p):
    """Build the known rank-18 Strassen+schoolbook decomposition."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def _lm_optimize(U, V, W, T, n, m, p, R, max_iters=200, tol=1e-12, deadline=None):
    """
    Optimize U, V, W using Levenberg-Marquardt on the flattened parameter vector.
    """
    U0 = np.array(U, dtype=np.float64)
    V0 = np.array(V, dtype=np.float64)
    W0 = np.array(W, dtype=np.float64)
    
    # Flatten
    params = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    # Sizes
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    
    def _residuals(params):
        U_f = params[:size_U].reshape(n, m, R)
        V_f = params[size_U:size_U+size_V].reshape(m, p, R)
        W_f = params[size_U+size_V:].reshape(p, n, R)
        
        # Reconstruct tensor
        # T[u, v, w] = sum_r U[u, j, r] * V[j, k, r] * W[k, i, r] ? 
        # Wait, the indices in _build_target are:
        # u = i*m + j
        # v = j*p + k
        # w = k*n + i
        # So U is indexed by (i, j), V by (j, k), W by (k, i).
        # My reshape above:
        # U_f shape (n, m, R) -> U[i, j, r]
        # V_f shape (m, p, R) -> V[j, k, r]
        # W_f shape (p, n, R) -> W[k, i, r]
        
        # T is (n*m, m*p, p*n).
        # T_flat index: ((i*m+j) * (m*p) + (j*p+k)) * (p*n) + (k*n+i)
        
        # Let's compute the full tensor
        # einsum: 'ijr,jkr,kir->ijk' gives T[i, j, k]
        # But T is defined with u, v, w.
        # T[i, j, k] corresponds to u=i*m+j, v=j*p+k, w=k*n+i.
        # The target T array in _build_target has 1s at these positions.
        
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_f, V_f, W_f)
        
        # We need to compare T_calc with the target T.
        # Target T has shape (n*m, m*p, p*n).
        # T_calc has shape (n, m, p).
        # We can reshape T_calc to (n*m, m*p, p*n) by permuting/reshaping?
        # Actually, T[u, v, w] = T[i*m+j, j*p+k, k*n+i].
        # Let's just create a target array of shape (n, m, p) for comparison in LM.
        # Target T_ijk[i, j, k] = 1 if the term exists.
        # In matrix multiplication C = A * B:
        # C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor decomposes the multiplication.
        # T[i, j, k] is 1 if the term A[i, j] * B[j, k] contributes to C[i, k].
        # Yes, so T_target[i, j, k] = 1 for all i, j, k.
        # Wait, is it?
        # The task says "exactly decompose the matrix multiplication tensor".
        # The standard matrix multiplication tensor has entries T[i, j, k] = 1 for all i, j, k in the respective dimensions?
        # Let's check _build_target.
        # It sets T[u, v, w] = 1.
        # u = i*m+j, v = j*p+k, w = k*n+i.
        # This is exactly the standard matmul tensor where every (i,j,k) triple has a 1.
        # So T_target[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
        
        T_target_local = np.ones((n, m, p), dtype=np.float64)
        
        res = (T_calc - T_target_local).flatten()
        return res

    def _jacobian(params):
        # Numerical Jacobian might be slow, but for small dims it's okay.
        # Or analytical.
        # Let's use numerical for simplicity and robustness, with central differences.
        # Dim of params: 4*R + 10*R + 10*R = 24*R. For R=17, ~400 params.
        # Residuals: n*m*p = 20.
        # J is 20x400.
        # Central diff: 2 * 400 evaluations of residual.
        # Residual eval is fast (einsum).
        
        eps = 1e-7
        J = np.zeros((n*m*p, len(params)))
        f0 = _residuals(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = _residuals(params_plus)
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = _residuals(params_minus)
            J[:, i] = (f_plus - f_minus) / (2 * eps)
        return J

    # Levenberg-Marquardt
    lambda_lm = 1e-3
    nu = 2.0
    best_err = np.inf
    best_params = params.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        f = _residuals(params)
        err = np.sum(f**2)
        
        if err < best_err:
            best_err = err
            best_params = params.copy()
            
        if best_err < 1e-20:
            break
            
        J = _jacobian(params)
        
        # Solve (J^T J + lambda * diag(J^T J)) delta = -J^T f
        JtJ = J.T @ J
        Jtf = J.T @ f
        
        # Use diagonal approximation for Hessian update (simple LM)
        # Or use full Hessian if we had it. We only have J.
        # Standard LM uses J^T J.
        
        improved = False
        for _ in range(10): # Inner loop for lambda adjustment
            try:
                H = JtJ + lambda_lm * np.diag(np.diag(JtJ))
                delta = np.linalg.solve(H, -Jtf)
                
                params_new = params + delta
                f_new = _residuals(params_new)
                err_new = np.sum(f_new**2)
                
                if err_new < err:
                    params = params_new
                    lambda_lm /= nu
                    nu = 2.0
                    improved = True
                    break
                else:
                    lambda_lm *= nu
                    nu *= 2.0
            except np.linalg.LinAlgError:
                lambda_lm *= nu
                nu *= 2.0
        
        if not improved:
            # If lambda gets too big, we might fail.
            if lambda_lm > 1e12:
                break

    # Extract best params
    U_best = best_params[:size_U].reshape(n, m, R)
    V_best = best_params[size_U:size_U+size_V].reshape(m, p, R)
    W_best = best_params[size_U+size_V:].reshape(p, n, R)
    
    # Convert back to list of lists format expected by _check_decomposition
    # U: (n*m, R)
    U_list = [[0]*R for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            u_idx = i*m + j
            for r in range(R):
                U_list[u_idx][r] = U_best[i, j, r]
                
    V_list = [[0]*R for _ in range(m*p)]
    for j in range(m):
        for k in range(p):
            v_idx = j*p + k
            for r in range(R):
                V_list[v_idx][r] = V_best[j, k, r]
                
    W_list = [[0]*R for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            w_idx = k*n + i
            for r in range(R):
                W_list[w_idx][r] = W_best[k, i, r]
                
    return U_list, V_list, W_list, best_err

def _search_rank(n, m, p, T, R, deadline, U18, V18, W18):
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    if R >= 18:
        return None
        
    terms_to_remove = 18 - R
    
    # Strategy: Remove specific terms from rank 18 and optimize the rest.
    # Schoolbook terms (14-17) are likely the easiest to remove if Strassen covers the rest.
    # Strassen terms are 0-13.
    
    # Initialize by taking rank 18, zeroing out some columns, and running LM.
    # We try different sets of columns to keep.
    
    # Heuristic: Keep Strassen terms, drop Schoolbook.
    # If R=17, drop 1 schoolbook term.
    # If R=16, drop 2 schoolbook terms.
    
    keep_indices_options = []
    
    # Option 1: Drop last schoolbook terms
    if terms_to_remove <= 4:
        keep_indices_options.append(list(range(14)) + list(range(14, 18 - terms_to_remove)))
        
    # Option 2: Drop random terms
    for seed in range(20):
        if time.time() > deadline:
            return None
        np.random.seed(seed + 5000)
        keep_indices = np.random.choice(18, R, replace=False)
        keep_indices_options.append(keep_indices.tolist())
        
    best_result = None
    best_err = np.inf
    
    for keep_indices in keep_indices_options:
        if time.time() > deadline:
            break
            
        U_init = U18_a[:, keep_indices].copy()
        V_init = V18_a[:, keep_indices].copy()
        W_init = W18_a[:, keep_indices].copy()
        
        # Add small noise to avoid exact zero gradients if stuck
        np.random.seed(42)
        noise_scale = 1e-4
        U_init += np.random.randn(4, R) * noise_scale
        V_init += np.random.randn(10, R) * noise_scale
        W_init += np.random.randn(10, R) * noise_scale
        
        U_opt, V_opt, W_opt, err = _lm_optimize(U_init, V_init, W_init, T, n, m, p, R, max_iters=100, deadline=deadline)
        
        if err < best_err:
            best_err = err
            best_result = (U_opt, V_opt, W_opt)
            
        if best_err < 1e-10:
            break
            
    if best_result is not None and best_err < 1e-10:
        U, V, W = best_result
        # Snap to rational
        for denom in [2, 4, 8, 16, 32, 64, 128, 256]:
            U_s = np.round(np.array(U) * denom) / denom
            V_s = np.round(np.array(V) * denom) / denom
            W_s = np.round(np.array(W) * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U, V, W
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    T = _build_target()
    
    U18, V18, W18 = _build_rank18(n, m, p)
    
    if not _check_decomposition(U18, V18, W18, T):
        return {"U": U18, "V": V18, "W": W18}

    # Try rank 17
    result17 = _search_rank(n, m, p, T, 17, deadline, U18, V18, W18)
    if result17 is not None:
        U, V, W = result17
        if _check_decomposition(U, V, W, T):
            return {"U": U, "V": V, "W": W}

    # Try rank 16
    result16 = _search_rank(n, m, p, T, 16, deadline, U18, V18, W18)
    if result16 is not None:
        U, V, W = result16
        if _check_decomposition(U, V, W, T):
            return {"U": U, "V": V, "W": W}

    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
