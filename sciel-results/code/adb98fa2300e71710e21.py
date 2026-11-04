import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    trivial_rank = n * m * p  # 24
    
    # Trivial solution for 2x3x4
    U_trivial = np.zeros((nm, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((mp, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((pn, trivial_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
                
    best_rank = trivial_rank
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial

    # Precompute the indices of the non-zero entries of the multiplication tensor
    # T[alpha, beta, gamma] = 1 if consistent, 0 otherwise
    # We only need to check the consistent ones for the "1" constraint
    # And implicitly the others are 0, but minimizing the sum of squares of errors
    # over all 576 entries is expensive.
    # However, the "0" constraints are important to prevent leakage.
    
    # Consistent indices:
    # i in 0..1, j in 0..2, k in 0..3
    # alpha = i*3 + j
    # beta = j*4 + k
    # gamma = k*2 + i
    
    # We will use a vectorized error function.
    # To speed up, we can precompute the indices.
    
    def get_error(U, V, W, rank):
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Compute T = sum_r U[:,r] V[:,r] W[:,r]
        # This is a tensor of shape (nm, mp, pn)
        # We can compute this efficiently using einsum or loops over rank
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 6x12x8 tensor. 576 elements.
        # Then we compare with the target tensor.
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target tensor is sparse.
        # Instead of creating a full 576 target tensor, we can just sum the errors
        # for the 24 non-zero entries and the 552 zero entries.
        # But creating the full tensor is fast enough (576 floats).
        
        # Let's create the target tensor once outside?
        # It depends on n,m,p which are fixed.
        
        # For now, compute error on non-zero entries
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    err += (T[a, b, c] - 1.0)**2
                    
        # Also add penalty for non-zero entries being non-zero?
        # The einsum creates a dense tensor.
        # We should penalize the other entries too.
        # T_flat = T.flatten()
        # Target_flat is 0 except at 24 indices.
        # We can construct the target flat vector.
        
        return err

    # Better: Precompute the target vector and indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = a * (mp * pn) + b * pn + c
                target_indices.append(idx)
                
    target_indices = np.array(target_indices)
    
    def error_func(x, rank):
        U = x[:nm*rank].reshape(nm, rank)
        V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = x[nm*rank+mp*rank:].reshape(pn, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_flat = T.flatten()
        
        # Error on non-zero entries
        err_nonzero = np.sum((T_flat[target_indices] - 1.0)**2)
        
        # Error on zero entries
        # We need indices of all entries NOT in target_indices
        # Total 576. 24 are nonzero. 552 are zero.
        # Creating a mask every time is slow.
        # Precompute zero_indices
        all_indices = np.arange(nm * mp * pn)
        zero_mask = np.ones(len(all_indices), dtype=bool)
        zero_mask[target_indices] = False
        zero_indices = all_indices[zero_mask]
        
        err_zero = np.sum(T_flat[zero_indices]**2)
        
        return err_nonzero + err_zero

    # Precompute zero_indices
    all_indices = np.arange(nm * mp * pn)
    zero_mask = np.ones(len(all_indices), dtype=bool)
    zero_mask[target_indices] = False
    zero_indices = all_indices[zero_mask]

    def error_func_optimized(x, rank):
        U = x[:nm*rank].reshape(nm, rank)
        V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = x[nm*rank+mp*rank:].reshape(pn, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_flat = T.flatten()
        
        err_nonzero = np.sum((T_flat[target_indices] - 1.0)**2)
        err_zero = np.sum(T_flat[zero_indices]**2)
        
        return err_nonzero + err_zero

    def check_valid(U, V, W, rank, tol=1e-6):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_flat = T.flatten()
        
        if np.max(np.abs(T_flat[target_indices] - 1.0)) > tol:
            return False
        if np.max(np.abs(T_flat[zero_indices])) > tol:
            return False
        return True

    # Try ranks from 16 to 23
    for rank in range(16, 24):
        if time.time() > deadline - 20.0:
            break
            
        # Multiple random restarts
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
                
            np.random.seed(42 + attempt)
            U_init = np.random.rand(nm, rank) * 0.5 - 0.25
            V_init = np.random.rand(mp, rank) * 0.5 - 0.25
            W_init = np.random.rand(pn, rank) * 0.5 - 0.25
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(lambda x: error_func_optimized(x, rank), x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 5000})
                if res.fun < 1e-8:
                    U_sol = res.x[:nm*rank].reshape(nm, rank)
                    V_sol = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_sol = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Snap to half-integers?
                    # The problem allows integer/half-integer.
                    # If the solution is close to half-integers, snap and check.
                    U_s = np.round(U_sol * 2) / 2.0
                    V_s = np.round(V_sol * 2) / 2.0
                    W_s = np.round(W_sol * 2) / 2.0
                    
                    if check_valid(U_s, V_s, W_s, rank):
                        best_rank = rank
                        best_U = U_s
                        best_V = V_s
                        best_W = W_s
                        break # Found valid for this rank, try lower rank
                    else:
                        # Try without snapping if snapping failed?
                        # The problem requires integer/half-integer.
                        # So we MUST snap.
                        pass
            except Exception:
                pass
                
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
