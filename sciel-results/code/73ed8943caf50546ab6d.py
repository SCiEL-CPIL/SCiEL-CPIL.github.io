import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

def _build_tensor(n, m, p):
    """
    Build the multiplication tensor T of shape (n*m, m*p, p*n).
    T[i, j, k] = 1 if i, j, k correspond to a valid multiplication term contributing to output k.
    
    Indexing:
    i = r * m + c   (A row r, col c)
    j = c' * p + k' (B row c', col k')
    k = k'' * n + r'' (C row r'', col k'')
    
    For standard matrix multiplication:
    C_{r, k''} = sum_{c} A_{r, c} B_{c, k''}
    
    So, a term A_{r, c} B_{c, k''} contributes to C_{r, k''}.
    i = r*m + c
    j = c*p + k''
    k = k''*n + r
    
    T[i, j, k] = 1
    """
    T = np.zeros((n * m, m * p, p * n))
    for r in range(n):
        for c in range(m):
            for k in range(p):
                i = r * m + c
                j = c * p + k
                kk = k * n + r
                T[i, j, kk] = 1.0
    return T

def _strassen_decomposition(n, m, p):
    """
    Construct a rank-28 decomposition for 2x4x4 using heterogeneous block structure.
    """
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Standard Strassen 2x2 patterns
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col_idx = 0
    for a_off, b_r_off, b_c_off, c_off in blocks:
        for pat in patterns:
            for local_i, coeff in pat['a'].items():
                r = local_i // 2
                c = local_i % 2
                global_i = r * m + (a_off + c)
                U[global_i, col_idx] = coeff
            
            for local_j, coeff in pat['b'].items():
                r = local_j // 2
                c = local_j % 2
                global_j = (b_r_off + r) * p + (b_c_off + c)
                V[global_j, col_idx] = coeff
            
            for local_k, coeff in pat['c'].items():
                r = local_k // 2
                c = local_k % 2
                global_k = c_off + c
                global_i = r
                W[global_k * n + global_i, col_idx] = coeff
            
            col_idx += 1
            
    return U, V, W

def _try_optimize_rank(T, n, m, p, target_rank, seed=0, U_init=None, V_init=None, W_init=None, time_limit=10.0):
    """
    Attempt to find a decomposition of rank 'target_rank' for tensor T.
    Uses alternating minimization with multiple restarts.
    Returns (U, V, W) if found with low error, else None.
    """
    start_time = time.time()
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    T_flat = T.flatten()
    
    def compute_residual_norm(U, V, W):
        T_calc = np.einsum('it,jt,kt->ijk', U, V, W)
        return np.linalg.norm(T_calc - T)
    
    def verify(U, V, W, atol=1e-9):
        T_check = np.einsum('it,jt,kt->ijk', U, V, W)
        return np.allclose(T_check, T, atol=atol)
    
    # Strategy: Alternating Minimization
    # We initialize U, V, W and then iteratively update one matrix at a time
    # while holding the others fixed, solving a linear least squares problem.
    
    max_iters = 500
    tol = 1e-12
    
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(seed + attempt)
        
        # Initialization
        if U_init is not None and attempt == 0:
            # Use first target_rank columns of Strassen
            U = U_init[:, :target_rank].copy()
            V = V_init[:, :target_rank].copy()
            W = W_init[:, :target_rank].copy()
            # Add small noise to escape symmetry
            U += np.random.randn(dim_u, target_rank) * 1e-5
            V += np.random.randn(dim_v, target_rank) * 1e-5
            W += np.random.randn(dim_w, target_rank) * 1e-5
        else:
            # Random initialization
            U = np.random.randn(dim_u, target_rank)
            V = np.random.randn(dim_v, target_rank)
            W = np.random.randn(dim_w, target_rank)
            
        prev_norm = np.inf
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Update U: Fix V, W. T_ij_k = sum_t U_i_t V_j_t W_k_t
            # For fixed V, W, we want to find U such that U V^T W^T approx T.
            # Let M = W^T T V. Then U M^T approx T? No.
            # T_{ijk} = sum_t U_{it} (V_{jt} W_{kt})
            # Reshape T to (dim_u, dim_v * dim_w).
            # T_reshaped[i, j*w + k] = sum_t U[i, t] * (V[:, t] * W[:, t])^T flattened?
            # Let Z_t = V[:, t] * W[:, t] (element-wise product, size dim_v*dim_w? No, V is dim_v x rank, W is dim_w x rank.
            # We need to form a matrix B of size (dim_v * dim_w) x rank where B_{j*w+k, t} = V_{j,t} * W_{k,t}.
            # Then T_reshaped (dim_u x dim_v*dim_w) = U (dim_u x rank) * B^T (rank x dim_v*dim_w).
            # So U = T_reshaped * (B^T B)^{-1} B^T.
            
            # Construct B
            # B[j, k, t] = V[j, t] * W[k, t]
            # Flatten j, k into single index
            B = np.einsum('jt,kt->jkt', V, W)
            B_flat = B.reshape(dim_v * dim_w, target_rank)
            
            T_reshaped = T.reshape(dim_u, dim_v * dim_w)
            
            # Solve U * B_flat^T = T_reshaped
            # U = T_reshaped * pinv(B_flat^T)
            try:
                U_new = np.linalg.lstsq(B_flat.T, T_reshaped.T, rcond=None)[0].T
            except:
                break
                
            # Update V
            # T_{ijk} = sum_t (U_{it} W_{kt}) V_{jt}
            # Let A_t = U[:, t] * W[:, t]
            # T_reshaped (dim_v, dim_u * dim_w) = V (dim_v x rank) * A^T
            A = np.einsum('it,kt->ikt', U_new, W)
            A_flat = A.reshape(dim_u * dim_w, target_rank)
            T_reshaped_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
            
            try:
                V_new = np.linalg.lstsq(A_flat.T, T_reshaped_v.T, rcond=None)[0].T
            except:
                break
                
            # Update W
            # T_{ijk} = sum_t (U_{it} V_{jt}) W_{kt}
            # Let C_t = U[:, t] * V[:, t]
            # T_reshaped (dim_w, dim_u * dim_v) = W (dim_w x rank) * C^T
            C = np.einsum('it,jt->ijt', U_new, V_new)
            C_flat = C.reshape(dim_u * dim_v, target_rank)
            T_reshaped_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
            
            try:
                W_new = np.linalg.lstsq(C_flat.T, T_reshaped_w.T, rcond=None)[0].T
            except:
                break
                
            U, V, W = U_new, V_new, W_new
            current_norm = compute_residual_norm(U, V, W)
            
            if np.abs(prev_norm - current_norm) < tol:
                if current_norm < 1e-9:
                    return U, V, W
                break
            prev_norm = current_norm
            
            if current_norm < 1e-9:
                return U, V, W
                
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 120.0  # Safety margin
    
    # 1. Get the known rank-28 solution
    U28, V28, W28 = _strassen_decomposition(n, m, p)
    T = _build_tensor(n, m, p)
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # 2. Try to reduce rank
    # We try ranks from 27 down to 16
    target_ranks = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:  # Keep 30s safety margin
            break
            
        # Allocate time based on rank difficulty (lower rank is harder)
        # But we have limited time. Let's give 10s per rank attempt initially.
        # If we find a solution, we can try lower.
        
        found = False
        # Try a few seeds for this rank
        for seed in range(3):
            if time.time() > deadline - 25:
                break
                
            res = _try_optimize_rank(T, n, m, p, target_rank, seed=seed,
                                     U_init=U28, V_init=V28, W_init=W28, time_limit=8.0)
            if res is not None:
                U_sol, V_sol, W_sol = res
                
                # Try to snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                T_check = np.einsum('it,jt,kt->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_check, T, atol=1e-9):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Try without snapping (if it's already close enough)
                    if np.allclose(np.einsum('it,jt,kt->ijk', U_sol, V_sol, W_sol), T, atol=1e-9):
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            # Found a solution, continue to try lower ranks if time permits
            if time.time() > deadline - 10:
                break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    res = construct()
    print("Rank used:", len(res['U'][0]))
