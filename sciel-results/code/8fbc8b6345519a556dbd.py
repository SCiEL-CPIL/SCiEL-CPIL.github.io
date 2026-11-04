import time
import numpy as np

_EVAL_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _EVAL_DEADLINE - 5.0 - time.time()

def _build_tensor(n, m, p):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = np.zeros((r_dim, i_dim, j_dim))
    for k in range(p):
        for i in range(n):
            for j in range(m):
                r = k * n + i
                jj = j * p + k
                T[r, i, jj] = 1.0
    return T

def _construct_rank50(n=2, m=4, p=7):
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M2
        U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        # M3
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M4
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
        col += 1
        # M5
        U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
        col += 1
        # M6
        U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M7
        U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 2x4x4: four 2x2x2 Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    # 2x4x2: two 2x2x2 Strassen blocks  
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    # 2x4x1: 8 direct terms
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1.0
            V[j*p + 6, col] = 1.0
            W[6*n + i, col] = 1.0
            col += 1

    assert col == rank
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = _build_tensor(n, m, p)
    T_mat = T.reshape(r_dim, i_dim * j_dim)
    rank = W.shape[1]
    cand = np.zeros_like(T_mat)
    for t in range(rank):
        wt = W[:, t]
        ut = U[:, t]
        vt = V[:, t]
        cand += np.outer(wt, np.outer(ut, vt).flatten())
    return np.max(np.abs(cand - T_mat)) < tol

def _try_reduce_rank(U, V, W, n, m, p, time_budget=30.0):
    """Try to reduce rank by 1 via local search."""
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    rank = W.shape[1]
    
    T = _build_tensor(n, m, p)
    T_mat = T.reshape(r_dim, i_dim * j_dim)
    
    rng = np.random.RandomState(12345)
    
    # Try removing each term and seeing if we can compensate
    for t_remove in range(rank):
        if _time_left() < 5:
            break
        
        # Remove term t_remove
        keep = [t for t in range(rank) if t != t_remove]
        U_keep = U[:, keep]
        V_keep = V[:, keep]
        W_keep = W[:, keep]
        
        # Compute residual from removed term
        residual = np.outer(W[:, t_remove], np.outer(U[:, t_remove], V[:, t_remove]).flatten())
        
        # Try to absorb this residual into existing terms by modifying them
        # Or try to find a new single term that covers it
        # For simplicity, try random modifications
        
        best_solution = None
        for attempt in range(500):
            if _time_left() < 5:
                break
            
            # Try modifying one of the kept terms
            t_mod = keep[rng.randint(0, len(keep))]
            
            U_new = U_keep.copy()
            V_new = V_keep.copy()
            W_new = W_keep.copy()
            
            # Add random perturbation to term t_mod
            # The perturbation should help cover the residual
            # residual = w_t * (u_t ⊗ v_t)
            # We want to find delta_w, delta_u, delta_v such that
            # (w_t + delta_w) * ((u_t + delta_u) ⊗ (v_t + delta_v)) ≈ w_t*(u_t⊗v_t) + residual
            # This is complex, so let's try a different approach
            
            # Alternative: try to find a new term that exactly covers the residual
            # residual is a matrix of shape (r_dim, i_dim*j_dim)
            # We need w_new, u_new, v_new such that outer(w_new, outer(u_new, v_new).flatten()) = residual
            
            # Check if residual is rank-1
            # SVD
            U_svd, s, Vt_svd = np.linalg.svd(residual, full_matrices=False)
            if s[1] < 1e-10:  # rank 1
                # Extract the rank-1 factors
                w_new = U_svd[:, 0] * np.sqrt(s[0])
                uv_flat = Vt_svd[0, :] * np.sqrt(s[0])
                uv = uv_flat.reshape(i_dim, j_dim)
                # Now factor uv = u_new * v_new^T
                U2_svd, s2, V2t_svd = np.linalg.svd(uv, full_matrices=False)
                if s2[1] < 1e-10:
                    u_new = U2_svd[:, 0] * np.sqrt(s2[0])
                    v_new = V2t_svd[0, :] * np.sqrt(s2[0])
                    
                    # Check if values are half-integers
                    def is_half_int(x, tol=1e-9):
                        r = 2 * x
                        return abs(r - round(r)) < tol
                    
                    if all(is_half_int(v) for v in w_new) and \
                       all(is_half_int(v) for v in u_new) and \
                       all(is_half_int(v) for v in v_new):
                        # Build the new solution
                        U_final = np.column_stack([U_new, u_new])
                        V_final = np.column_stack([V_new, v_new])
                        W_final = np.column_stack([W_new, w_new])
                        
                        # Verify
                        cand = np.zeros_like(T_mat)
                        for t in range(rank - 1):
                            wt = W_final[:, t]
                            ut = U_final[:, t]
                            vt = V_final[:, t]
                            cand += np.outer(wt, np.outer(ut, vt).flatten())
                        
                        if np.max(np.abs(cand - T_mat)) < 1e-10:
                            best_solution = (U_final, V_final, W_final)
                            break
        
        if best_solution is not None:
            return best_solution
    
    return None

def _greedy_improve(U, V, W, n, m, p, time_budget=20.0):
    """Try to improve the rank via greedy local search."""
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    rank = W.shape[1]
    
    T = _build_tensor(n, m, p)
    T_mat = T.reshape(r_dim, i_dim * j_dim)
    
    rng = np.random.RandomState(99999)
    
    # Try swapping pairs of terms
    for iter in range(1000):
        if _time_left() < 5:
            break
        
        t1 = rng.randint(0, rank)
        t2 = rng.randint(0, rank)
        if t1 == t2:
            continue
        
        # Try to merge terms t1 and t2 into a single term
        u1, v1, w1 = U[:, t1], V[:, t1], W[:, t1]
        u2, v2, w2 = U[:, t2], V[:, t2], W[:, t2]
        
        combined = np.outer(w1, np.outer(u1, v1).flatten()) + \
                   np.outer(w2, np.outer(u2, v2).flatten())
        
        # Check if combined is rank-1
        U_svd, s, Vt_svd = np.linalg.svd(combined, full_matrices=False)
        if len(s) > 1 and s[1] < 1e-10:
            w_new = U_svd[:, 0] * np.sqrt(s[0])
            uv_flat = Vt_svd[0, :] * np.sqrt(s[0])
            uv = uv_flat.reshape(i_dim, j_dim)
            U2_svd, s2, V2t_svd = np.linalg.svd(uv, full_matrices=False)
            if len(s2) > 1 and s2[1] < 1e-10:
                u_new = U2_svd[:, 0] * np.sqrt(s2[0])
                v_new = V2t_svd[0, :] * np.sqrt(s2[0])
                
                def is_half_int(x, tol=1e-9):
                    r = 2 * x
                    return abs(r - round(r)) < tol
                
                if all(is_half_int(v) for v in w_new) and \
                   all(is_half_int(v) for v in u_new) and \
                   all(is_half_int(v) for v in v_new):
                    # Build new solution with rank-1
                    keep = [t for t in range(rank) if t not in (t1, t2)]
                    U_new = np.column_stack([U[:, keep], u_new])
                    V_new = np.column_stack([V[:, keep], v_new])
                    W_new = np.column_stack([W[:, keep], w_new])
                    
                    # Verify
                    cand = np.zeros_like(T_mat)
                    for t in range(rank - 1):
                        wt = W_new[:, t]
                        ut = U_new[:, t]
                        vt = V_new[:, t]
                        cand += np.outer(wt, np.outer(ut, vt).flatten())
                    
                    if np.max(np.abs(cand - T_mat)) < 1e-10:
                        return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = 2, 4, 7
    
    # Construct rank-50 baseline
    U, V, W = _construct_rank50(n, m, p)
    
    # Verify baseline
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Baseline verification failed")
    
    # Try to improve to rank 49
    if _time_left() > 10:
        result = _try_reduce_rank(U, V, W, n, m, p, min(_time_left(), 40.0))
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
    
    # Try greedy improvement
    if _time_left() > 10:
        result = _greedy_improve(U, V, W, n, m, p, min(_time_left(), 30.0))
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
    
    # Convert to list format
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    rank = W.shape[1]
    
    U_list = [[0.0] * rank for _ in range(i_dim)]
    V_list = [[0.0] * rank for _ in range(j_dim)]
    W_list = [[0.0] * rank for _ in range(r_dim)]
    
    for i in range(i_dim):
        for t in range(rank):
            U_list[i][t] = float(U[i, t])
    for j in range(j_dim):
        for t in range(rank):
            V_list[j][t] = float(V[j, t])
    for r in range(r_dim):
        for t in range(rank):
            W_list[r][t] = float(W[r, t])
    
    return {"U": U_list, "V": V_list, "W": W_list}
