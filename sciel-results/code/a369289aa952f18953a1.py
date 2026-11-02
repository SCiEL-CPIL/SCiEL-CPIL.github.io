"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition.

Uses iterative rank-1 peeling with linear algebra to find low-rank exact decompositions.
"""

import time
import numpy as np

DEADLINE_S = 110.0
DIMENSIONS = [4, 5, 5]


def _build_tensor_3way(n, m, p):
    """Build the multiplication tensor T[i, j, k] = 1 for all (i,j,k)."""
    return np.ones((n, m, p), dtype=np.float64)


def _validate_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Validate rank-R decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _peel_rank1_component(residual, n, m, p, mode='i', seed=None):
    """Peel a rank-1 component from a 3-way tensor residual.
    
    Strategy: Fix a random vector in one mode, then solve for the other two.
    For mode 'i': pick u (n,), compute M[j,k] = sum_i residual[i,j,k]*u[i],
    then find rank-1 approximation of M as v*w^T via SVD.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    if mode == 'i':
        u = rng.randn(n)
        # M[j,k] = sum_i residual[i,j,k] * u[i]
        M = np.tensordot(u, residual, axes=[0, 0])  # shape (m, p)
        # SVD of M
        U_mat, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        if len(s_vals) == 0 or s_vals[0] < 1e-12:
            return None
        v = U_mat[:, 0] * np.sqrt(s_vals[0])  # (m,)
        w = Vt[0, :] * np.sqrt(s_vals[0])      # (p,)
        # Normalize: we want u*v*w to be the component
        # Actually, M = v*w^T approximately, so residual - outer(u,v,w)
        # But M[j,k] = sum_i residual[i,j,k]*u[i], and we approximated M ≈ v*w^T
        # So the component is outer(u, v, w)
        return u, v, w
    elif mode == 'j':
        v = rng.randn(m)
        M = np.tensordot(v, residual, axes=[0, 1])  # shape (n, p)
        U_mat, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        if len(s_vals) == 0 or s_vals[0] < 1e-12:
            return None
        u = U_mat[:, 0] * np.sqrt(s_vals[0])
        w = Vt[0, :] * np.sqrt(s_vals[0])
        return u, v, w
    elif mode == 'k':
        w = rng.randn(p)
        M = np.tensordot(w, residual, axes=[0, 2])  # shape (n, m)
        U_mat, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        if len(s_vals) == 0 or s_vals[0] < 1e-12:
            return None
        u = U_mat[:, 0] * np.sqrt(s_vals[0])
        v = Vt[0, :] * np.sqrt(s_vals[0])
        return u, v, w
    return None


def _try_peeling_decomposition(T, n, m, p, target_rank, max_restarts=20, tol=1e-8):
    """Try to find a rank-target_rank decomposition by iterative peeling."""
    t_start = time.time()
    
    for restart in range(max_restarts):
        if time.time() - t_start > DEADLINE_S - 25:
            break
        
        residual = T.copy()
        components = []
        success = True
        
        for r in range(target_rank):
            if time.time() - t_start > DEADLINE_S - 25:
                success = False
                break
            
            # Try multiple modes and random seeds
            best_component = None
            best_reduction = 0.0
            old_norm = np.sum(residual**2)
            
            for attempt in range(30):
                mode = ['i', 'j', 'k'][attempt % 3]
                comp = _peel_rank1_component(residual, n, m, p, mode=mode, seed=restart * 100 + attempt)
                if comp is None:
                    continue
                
                u, v, w = comp
                peeled = np.einsum('i,j,k->ijk', u, v, w)
                new_residual = residual - peeled
                new_norm = np.sum(new_residual**2)
                
                reduction = old_norm - new_norm
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_component = (u, v, w, new_residual)
            
            if best_component is None or best_reduction < 1e-10:
                success = False
                break
            
            u, v, w, residual = best_component
            components.append((u, v, w))
        
        if not success or len(components) != target_rank:
            continue
        
        # Check if residual is close to zero
        if np.max(np.abs(residual)) < tol:
            # Build the factor matrices
            U_full = np.zeros((n * m, target_rank))
            V_full = np.zeros((m * p, target_rank))
            W_full = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                u_r, v_r, w_r = components[r]
                for i in range(n):
                    for j in range(m):
                        U_full[i * m + j, r] = u_r[i]
                for j in range(m):
                    for k in range(p):
                        V_full[j * p + k, r] = v_r[j]
                for k in range(p):
                    for i in range(n):
                        W_full[k * n + i, r] = w_r[k]
            
            if _validate_decomposition(U_full, V_full, W_full, n, m, p, tol):
                return U_full, V_full, W_full
    
    return None


def _try_polynomial_peeling(T, n, m, p, target_rank, max_restarts=10, tol=1e-8):
    """Alternative: use the fact that for matrix mult, we can use structured peeling.
    
    For T[i,j,k]=1, a rank-1 component is outer(u,v,w).
    We can try to find components where u, v, w have sparse or structured support.
    """
    t_start = time.time()
    
    for restart in range(max_restarts):
        if time.time() - t_start > DEADLINE_S - 25:
            break
        
        residual = T.copy()
        components = []
        success = True
        
        for r in range(target_rank):
            if time.time() - t_start > DEADLINE_S - 25:
                success = False
                break
            
            # Try random sparse vectors
            best_component = None
            best_residual_norm = np.sum(residual**2)
            
            for attempt in range(50):
                # Random support patterns
                u = np.zeros(n)
                v = np.zeros(m)
                w = np.zeros(p)
                
                # Random non-zero entries
                for i in range(n):
                    if np.random.rand() < 0.7:
                        u[i] = np.random.choice([-1, 1]) * np.random.rand() + 0.1
                for j in range(m):
                    if np.random.rand() < 0.7:
                        v[j] = np.random.choice([-1, 1]) * np.random.rand() + 0.1
                for k in range(p):
                    if np.random.rand() < 0.7:
                        w[k] = np.random.choice([-1, 1]) * np.random.rand() + 0.1
                
                # Optimize w given u, v
                denom = np.sum(u**2) * np.sum(v**2)
                if abs(denom) < 1e-12:
                    continue
                
                A = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        A[i, k] = np.dot(residual[i, :, k], v)
                numerator = np.dot(u, A)
                w_opt = numerator / denom
                
                peeled = np.einsum('i,j,k->ijk', u, v, w_opt)
                new_residual = residual - peeled
                new_norm = np.sum(new_residual**2)
                
                if new_norm < best_residual_norm:
                    best_residual_norm = new_norm
                    best_component = (u, v, w_opt, new_residual)
            
            if best_component is None or best_residual_norm >= np.sum(residual**2) * 0.99:
                success = False
                break
            
            u, v, w, residual = best_component
            components.append((u, v, w))
        
        if not success or len(components) != target_rank:
            continue
        
        if np.max(np.abs(residual)) < tol:
            U_full = np.zeros((n * m, target_rank))
            V_full = np.zeros((m * p, target_rank))
            W_full = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                u_r, v_r, w_r = components[r]
                for i in range(n):
                    for j in range(m):
                        U_full[i * m + j, r] = u_r[i]
                for j in range(m):
                    for k in range(p):
                        V_full[j * p + k, r] = v_r[j]
                for k in range(p):
                    for i in range(n):
                        W_full[k * n + i, r] = w_r[k]
            
            if _validate_decomposition(U_full, V_full, W_full, n, m, p, tol):
                return U_full, V_full, W_full
    
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U, V, W)
    
    T = _build_tensor_3way(n, m, p)
    
    # Try ranks from low to high, stopping at first success
    # For 4x5x5, known lower bounds suggest rank around 60-80
    # Try from 60 up to 99
    
    for target_rank in range(60, best_rank):
        if time.time() - t0 > DEADLINE_S - 30:
            break
        
        # Try the SVD-based peeling
        result = _try_peeling_decomposition(T, n, m, p, target_rank, max_restarts=15, tol=1e-8)
        if result is not None:
            U_new, V_new, W_new = result
            best_rank = target_rank
            best_result = (U_new, V_new, W_new)
            break
        
        # Try polynomial peeling
        if time.time() - t0 > DEADLINE_S - 30:
            break
        result = _try_polynomial_peeling(T, n, m, p, target_rank, max_restarts=10, tol=1e-8)
        if result is not None:
            U_new, V_new, W_new = result
            best_rank = target_rank
            best_result = (U_new, V_new, W_new)
            break
    
    # Convert to lists with integer cleanup
    def _to_list(M):
        out = []
        for row in M:
            lrow = []
            for x in row:
                xv = float(x)
                if abs(xv - round(xv)) < 1e-6:
                    xv = int(round(xv))
                lrow.append(xv)
            out.append(lrow)
        return out
    
    U_best, V_best, W_best = best_result
    return {
        "U": _to_list(np.array(U_best)),
        "V": _to_list(np.array(V_best)),
        "W": _to_list(np.array(W_best))
    }
