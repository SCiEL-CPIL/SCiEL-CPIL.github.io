"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank_school = n * m * p
    
    # Schoolbook decomposition
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Pre-compute support: list of (i_idx, j_idx, k_idx)
    support = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i_idx = a*m + b
                j_idx = b*p + c
                k_idx = c*n + a
                support.append((i_idx, j_idx, k_idx))
    
    # Build support as numpy arrays for vectorized evaluation
    sup_i = np.array([s[0] for s in support])
    sup_j = np.array([s[1] for s in support])
    sup_k = np.array([s[2] for s in support])
    num_support = len(support)
    
    # All off-support points
    all_points = []
    for ii in range(nm):
        for jj in range(mp):
            for kk in range(pn):
                a, b = divmod(ii, m)
                b2, c = divmod(jj, p)
                c2, a2 = divmod(kk, n)
                if not (b == b2 and c == c2 and a == a2):
                    all_points.append((ii, jj, kk))
    off_support = np.array(all_points)
    off_i = off_support[:, 0]
    off_j = off_support[:, 1]
    off_k = off_support[:, 2]
    num_off = len(off_support)
    
    def evaluate_tensor_full(U_mat, V_mat, W_mat):
        """Evaluate tensor at all support and off-support points."""
        U_s = U_mat[sup_i, :]
        V_s = V_mat[sup_j, :]
        W_s = W_mat[sup_k, :]
        T_supp = np.sum(U_s * V_s * W_s, axis=1)
        
        U_o = U_mat[off_i, :]
        V_o = V_mat[off_j, :]
        W_o = W_mat[off_k, :]
        T_off = np.sum(U_o * V_o * W_o, axis=1)
        
        return T_supp, T_off
    
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if decomposition is valid. Returns True if valid."""
        T_supp, T_off = evaluate_tensor_full(U_mat, V_mat, W_mat)
        if np.max(np.abs(T_supp - 1.0)) > tol:
            return False
        if np.max(np.abs(T_off)) > tol:
            return False
        return True
    
    def objective_and_grad(U, V, W):
        """Compute objective and gradients for the full tensor fitting problem."""
        T_supp, T_off = evaluate_tensor_full(U, V, W)
        obj = np.sum((T_supp - 1.0)**2) + np.sum(T_off**2)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Support gradients
        err_s = T_supp - 1.0
        for idx in range(num_support):
            i_idx = sup_i[idx]
            j_idx = sup_j[idx]
            k_idx = sup_k[idx]
            e = err_s[idx]
            vw = V[j_idx, :] * W[k_idx, :]
            u_row = U[i_idx, :]
            v_row = V[j_idx, :]
            w_row = W[k_idx, :]
            grad_U[i_idx, :] += 2.0 * e * vw
            grad_V[j_idx, :] += 2.0 * e * (u_row * w_row)
            grad_W[k_idx, :] += 2.0 * e * (u_row * v_row)
        
        # Off-support gradients
        for idx in range(num_off):
            i_idx = off_i[idx]
            j_idx = off_j[idx]
            k_idx = off_k[idx]
            t_val = T_off[idx]
            if abs(t_val) < 1e-12:
                continue
            u_row = U[i_idx, :]
            v_row = V[j_idx, :]
            w_row = W[k_idx, :]
            grad_U[i_idx, :] += 2.0 * t_val * (v_row * w_row)
            grad_V[j_idx, :] += 2.0 * t_val * (u_row * w_row)
            grad_W[k_idx, :] += 2.0 * t_val * (u_row * v_row)
        
        return obj, grad_U, grad_V, grad_W
    
    def optimize_from_init(U_init, V_init, W_init, r, max_iter=200, tol=1e-12):
        """Optimize using L-BFGS-like method with line search."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        best_obj = np.inf
        patience = 0
        max_patience = 30
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            obj, grad_U, grad_V, grad_W = objective_and_grad(U, V, W)
            
            if obj < best_obj:
                best_obj = obj
                patience = 0
                U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            else:
                patience += 1
            
            if patience > max_patience:
                break
            
            if obj < tol:
                break
            
            # Backtracking line search
            lr = 0.1
            for _ in range(20):
                U_new = U - lr * grad_U
                V_new = V - lr * grad_V
                W_new = W - lr * grad_W
                obj_new, _, _, _ = objective_and_grad(U_new, V_new, W_new)
                if obj_new < obj * 0.999:
                    U, V, W = U_new, V_new, W_new
                    break
                lr *= 0.5
            
            # Check convergence
            if obj < tol:
                break
        
        return U, V, W, best_obj
    
    # Strategy: try to reduce rank by starting from subsets of schoolbook terms
    # and optimizing
    
    # Pre-compute schoolbook terms as vectors
    school_U = np.zeros((nm, rank_school))
    school_V = np.zeros((mp, rank_school))
    school_W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                school_U[i*m + j, col] = 1
                school_V[j*p + k, col] = 1
                school_W[k*n + i, col] = 1
                col += 1
    
    # Try ranks from rank_school-1 down
    for r in range(rank_school - 1, max(60, rank_school - 20), -1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Strategy 1: Start with r random schoolbook terms
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Pick r random schoolbook indices
            idxs = np.random.choice(rank_school, size=r, replace=False)
            U_try = school_U[:, idxs].copy()
            V_try = school_V[:, idxs].copy()
            W_try = school_W[:, idxs].copy()
            
            U_opt, V_opt, W_opt, obj = optimize_from_init(U_try, V_try, W_try, r, max_iter=100)
            
            if obj < 1e-10:
                if check_decomposition(U_opt, V_opt, W_opt, tol=1e-9):
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = r
                    found = True
                    break
        
        # Strategy 2: Start with all schoolbook terms but with some zeroed out
        if not found and time.time() < deadline:
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                # Zero out some random columns
                zero_mask = np.zeros(rank_school, dtype=bool)
                num_zero = rank_school - r
                zero_idx = np.random.choice(rank_school, size=num_zero, replace=False)
                zero_mask[zero_idx] = True
                
                U_try = school_U.copy()
                V_try = school_V.copy()
                W_try = school_W.copy()
                U_try[:, zero_mask] = 0
                V_try[:, zero_mask] = 0
                W_try[:, zero_mask] = 0
                
                # Now we have rank_school columns but some are zero, so effective rank is r
                # Remove zero columns
                nonzero_cols = ~zero_mask
                U_try = U_try[:, nonzero_cols]
                V_try = V_try[:, nonzero_cols]
                W_try = W_try[:, nonzero_cols]
                
                U_opt, V_opt, W_opt, obj = optimize_from_init(U_try, V_try, W_try, r, max_iter=100)
                
                if obj < 1e-10:
                    if check_decomposition(U_opt, V_opt, W_opt, tol=1e-9):
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        best_rank = r
                        found = True
                        break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
