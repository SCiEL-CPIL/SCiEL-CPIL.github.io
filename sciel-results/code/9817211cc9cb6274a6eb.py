"""Tensor decomposition search for 2x3 x 3x6 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify that U,V,W decompose T: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Compute sum_r U[:,r] * V[:,r] * W[:,r] as outer products
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(reconstructed, T, atol=tol)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(T, n, m, p, max_rank, time_limit=120, deadline_margin=10):
    """
    Try to find a decomposition with rank < max_rank using iterative local search.
    Uses randomized initialization and gradient-free local search on sparse structures.
    """
    start_time = time.time()
    deadline = start_time + time_limit - deadline_margin
    
    # We'll work with rational/integer-friendly representations
    # Try to find sparse decompositions
    
    best_result = None
    
    # Strategy: Start from schoolbook, try to eliminate terms
    # by finding linear dependencies among the rank-1 components
    
    U_sb = np.array(_schoolbook(n, m, p)["U"], dtype=np.float64)
    V_sb = np.array(_schoolbook(n, m, p)["V"], dtype=np.float64)
    W_sb = np.array(_schoolbook(n, m, p)["W"], dtype=np.float64)
    
    rank_sb = U_sb.shape[1]
    
    # For each target rank r from max_rank-1 down to max(1, max_rank - 5):
    for target_r in range(max_rank - 1, max(max_rank - 6, 1), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Random sparse initialization: pick target_r random rank-1 terms
            # with small integer entries, then refine
            U_init = np.zeros((n*m, target_r))
            V_init = np.zeros((m*p, target_r))
            W_init = np.zeros((p*n, target_r))
            
            # Seed with a subset of schoolbook terms plus some random perturbation
            indices = np.random.choice(rank_sb, size=min(target_r, rank_sb), replace=False)
            U_init[:, :len(indices)] = U_sb[:, indices]
            V_init[:, :len(indices)] = V_sb[:, indices]
            W_init[:, :len(indices)] = W_sb[:, indices]
            
            # If target_r < rank_sb, we've dropped some terms; try to adjust
            if target_r < rank_sb:
                # Add small random noise to remaining columns to break symmetry
                mask = np.random.random((n*m, target_r)) < 0.1
                U_init[mask] += np.random.randint(-1, 2, size=mask.shape)
                mask = np.random.random((m*p, target_r)) < 0.1
                V_init[mask] += np.random.randint(-1, 2, size=mask.shape)
                mask = np.random.random((p*n, target_r)) < 0.1
                W_init[mask] += np.random.randint(-1, 2, size=mask.shape)
            
            # Local search: iteratively adjust one entry at a time to reduce residual
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                # Compute residual
                residual = T - np.einsum('ar,br,cr->abc', U_init, V_init, W_init)
                err = np.abs(residual).max()
                
                if err < 1e-9:
                    # Found a valid decomposition!
                    # Round to integers or half-integers
                    U_rounded = np.round(U_init * 2) / 2
                    V_rounded = np.round(V_init * 2) / 2
                    W_rounded = np.round(W_init * 2) / 2
                    if _verify(U_rounded, V_rounded, W_rounded, T):
                        best_result = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        return best_result
                    # Try without rounding
                    best_result = {
                        "U": U_init.tolist(),
                        "V": V_init.tolist(),
                        "W": W_init.tolist()
                    }
                    return best_result
                
                # Pick a random element to modify
                r = np.random.randint(target_r)
                choice = np.random.randint(3)
                if choice == 0:
                    a = np.random.randint(n*m)
                    delta = np.random.choice([-1, 1]) * (1 if np.random.random() < 0.5 else 0.5)
                    old_val = U_init[a, r]
                    U_init[a, r] += delta
                    # Check if error decreased
                    new_residual = T - np.einsum('ar,br,cr->abc', U_init, V_init, W_init)
                    new_err = np.abs(new_residual).max()
                    if new_err > err + 1e-6:
                        U_init[a, r] = old_val
                elif choice == 1:
                    b = np.random.randint(m*p)
                    delta = np.random.choice([-1, 1]) * (1 if np.random.random() < 0.5 else 0.5)
                    old_val = V_init[b, r]
                    V_init[b, r] += delta
                    new_residual = T - np.einsum('ar,br,cr->abc', U_init, V_init, W_init)
                    new_err = np.abs(new_residual).max()
                    if new_err > err + 1e-6:
                        V_init[b, r] = old_val
                else:
                    c = np.random.randint(p*n)
                    delta = np.random.choice([-1, 1]) * (1 if np.random.random() < 0.5 else 0.5)
                    old_val = W_init[c, r]
                    W_init[c, r] += delta
                    new_residual = T - np.einsum('ar,br,cr->abc', U_init, V_init, W_init)
                    new_err = np.abs(new_residual).max()
                    if new_err > err + 1e-6:
                        W_init[c, r] = old_val
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    schoolbook_rank = n * m * p  # 36
    
    # First, get the guaranteed schoolbook solution
    result = _schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    start = time.time()
    deadline = start + 110  # Leave 10s margin before 120s limit
    
    # Try ranks from 35 down to 30
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 7, 1), -1):
        if time.time() > deadline:
            break
        
        better = _try_reduce_rank(T, n, m, p, target_rank + 1, 
                                  time_limit=110, 
                                  deadline_margin=time.time() - start)
        if better is not None:
            result = better
            break
    
    return result
