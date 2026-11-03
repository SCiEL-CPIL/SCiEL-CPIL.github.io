"""Matrix multiplication 3x3 x 3x7: rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly reproduces the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)
    and all other entries are 0.
    """
    rank = len(U[0]) if isinstance(U[0], list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T[a, b, c] = sum_d U[a,d] * V[b,d] * W[c,d]
    T = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
    
    # Check target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-8:
                    return False
    
    # Check that all other entries are 0
    valid = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid[a, b, c] = True
    
    invalid_mask = ~valid
    if np.any(np.abs(T[invalid_mask]) > 1e-8):
        return False
        
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # 1. Schoolbook (rank 63) as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a known low-rank decomposition via structured initialization
    # For matrix multiplication, there are known algorithms (Strassen-like)
    # that can achieve lower rank than schoolbook
    
    # Precompute target tensor T_target (n*m, m*p, p*n)
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    from scipy.optimize import minimize
    
    def snap(val):
        # Snap to nearest integer or half-integer
        nearest_int = round(val)
        nearest_half = round(val * 2) / 2.0
        if abs(val - nearest_int) < 1e-6:
            return float(nearest_int)
        if abs(val - nearest_half) < 1e-6:
            return float(nearest_half)
        return val
    
    def objective(x, r):
        U_flat = x[:n*m*r]
        V_flat = x[n*m*r:n*m*r + m*p*r]
        W_flat = x[n*m*r + m*p*r:]
        
        U_mat = U_flat.reshape(n*m, r)
        V_mat = V_flat.reshape(m*p, r)
        W_mat = W_flat.reshape(p*n, r)
        
        T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
        # Only compute error on valid entries for efficiency
        # But we need full tensor for gradient, so compute all
        return np.sum((T - T_target)**2)
    
    def objective_valid_only(x, r):
        """Compute objective only on valid entries for faster evaluation"""
        U_flat = x[:n*m*r]
        V_flat = x[n*m*r:n*m*r + m*p*r]
        W_flat = x[n*m*r + m*p*r:]
        
        U_mat = U_flat.reshape(n*m, r)
        V_mat = V_flat.reshape(m*p, r)
        W_mat = W_flat.reshape(p*n, r)
        
        T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
        
        # Sum squared error only on valid entries
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    err += (T[a, b, c] - 1.0) ** 2
        return err
    
    # Try ranks from 62 down to 35
    # Known lower bounds suggest optimal is around 40-50 for this shape
    for target_rank in range(62, 34, -1):
        if time.time() > deadline - 15.0:
            break
            
        found = False
        # Number of restarts based on remaining time
        remaining = deadline - time.time()
        max_restarts = min(25, max(5, int(remaining / 4.0)))
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize with random matrices, scaled appropriately
            # Use different initialization strategies
            if restart % 5 == 0:
                # Small random
                scale = 0.5 / np.sqrt(target_rank)
                U = np.random.randn(n*m, target_rank) * scale
                V = np.random.randn(m*p, target_rank) * scale
                W = np.random.randn(p*n, target_rank) * scale
            elif restart % 5 == 1:
                # Moderate random with bias
                scale = 1.0 / np.sqrt(target_rank)
                U = np.random.randn(n*m, target_rank) * scale + 0.5
                V = np.random.randn(m*p, target_rank) * scale + 0.5
                W = np.random.randn(p*n, target_rank) * scale + 0.5
            elif restart % 5 == 2:
                # Sparse-ish initialization
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                for c in range(target_rank):
                    U[np.random.randint(0, n*m), c] = 1.0
                    V[np.random.randint(0, m*p), c] = 1.0
                    W[np.random.randint(0, p*n), c] = 1.0
            elif restart % 5 == 3:
                # Structured: try to mimic Strassen-like patterns
                # Initialize with blocks of 1s and -1s
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                for c in range(target_rank):
                    # Random sign pattern
                    signs = np.random.choice([-1.0, 1.0], size=n*m)
                    U[:, c] = signs * 0.5
                    signs = np.random.choice([-1.0, 1.0], size=m*p)
                    V[:, c] = signs * 0.5
                    signs = np.random.choice([-1.0, 1.0], size=p*n)
                    W[:, c] = signs * 0.5
            else:
                # Very sparse: only a few non-zeros per column
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                for c in range(target_rank):
                    nz_u = np.random.randint(1, 4)
                    nz_v = np.random.randint(1, 4)
                    nz_w = np.random.randint(1, 4)
                    idx_u = np.random.choice(n*m, nz_u, replace=False)
                    idx_v = np.random.choice(m*p, nz_v, replace=False)
                    idx_w = np.random.choice(p*n, nz_w, replace=False)
                    U[idx_u, c] = np.random.choice([-1.0, 1.0], nz_u)
                    V[idx_v, c] = np.random.choice([-1.0, 1.0], nz_v)
                    W[idx_w, c] = np.random.choice([-1.0, 1.0], nz_w)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B with adaptive iteration limits
            max_iter = min(800, max(150, int((deadline - time.time()) * 15)))
            try:
                res = minimize(objective, x0, args=(target_rank,), method='L-BFGS-B', 
                               options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 
                                       'maxfun': max_iter * 10, 'maxls': 50})
            except Exception:
                continue
            
            # Check if we got close to zero error
            if res.fun < 1e-8:
                # Check validity
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Convert to lists of floats
                U_list = [[float(x) for x in row] for row in U_opt]
                V_list = [[float(x) for x in row] for row in V_opt]
                W_list = [[float(x) for x in row] for row in W_opt]
                
                # First try snapping to integers
                U_int = [[float(round(v)) for v in row] for row in U_list]
                V_int = [[float(round(v)) for v in row] for row in V_list]
                W_int = [[float(round(v)) for v in row] for row in W_list]
                
                if _verify(U_int, V_int, W_int, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_int, V_int, W_int
                    found = True
                    break
                
                # Then try snapping to half-integers
                U_snap = [[snap(v) for v in row] for row in U_list]
                V_snap = [[snap(v) for v in row] for row in V_list]
                W_snap = [[snap(v) for v in row] for row in W_list]
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    found = True
                    break
                elif _verify(U_list, V_list, W_list, n, m, p):
                    # Valid but not snapped - accept if it verifies
                    best_rank = target_rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    found = True
                    break
                    
        if found:
            # Continue searching for lower ranks
            pass
    
    # Final polish: if we found a solution, try to verify it once more
    if best_rank < rank_sb:
        if not _verify(best_U, best_V, best_W, n, m, p):
            # Fallback to schoolbook
            best_rank = rank_sb
            best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # If we have time, try to reduce rank further from best found
    if best_rank < rank_sb and time.time() < deadline - 30.0:
        # Try to reduce by 1-2 more ranks
        for extra_reduction in range(1, 3):
            target_rank2 = best_rank - extra_reduction
            if target_rank2 < 35:
                break
            if time.time() > deadline - 10.0:
                break
            
            for restart in range(5):
                if time.time() > deadline - 5.0:
                    break
                
                # Initialize near the previous solution's structure
                U = np.random.randn(n*m, target_rank2) * 0.3
                V = np.random.randn(m*p, target_rank2) * 0.3
                W = np.random.randn(p*n, target_rank2) * 0.3
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                max_iter = min(300, max(50, int((deadline - time.time()) * 8)))
                try:
                    res = minimize(objective, x0, args=(target_rank2,), method='L-BFGS-B', 
                                   options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 
                                           'maxfun': max_iter * 10})
                except Exception:
                    continue
                
                if res.fun < 1e-8:
                    U_opt = res.x[:n*m*target_rank2].reshape(n*m, target_rank2)
                    V_opt = res.x[n*m*target_rank2:n*m*target_rank2 + m*p*target_rank2].reshape(m*p, target_rank2)
                    W_opt = res.x[n*m*target_rank2 + m*p*target_rank2:].reshape(p*n, target_rank2)
                    
                    U_list = [[float(x) for x in row] for row in U_opt]
                    V_list = [[float(x) for x in row] for row in V_opt]
                    W_list = [[float(x) for x in row] for row in W_opt]
                    
                    U_int = [[float(round(v)) for v in row] for row in U_list]
                    V_int = [[float(round(v)) for v in row] for row in V_list]
                    W_int = [[float(round(v)) for v in row] for row in W_list]
                    
                    if _verify(U_int, V_int, W_int, n, m, p):
                        best_rank = target_rank2
                        best_U, best_V, best_W = U_int, V_int, W_int
                        break
                    elif _verify(U_list, V_list, W_list, n, m, p):
                        best_rank = target_rank2
                        best_U, best_V, best_W = U_list, V_list, W_list
                        break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
