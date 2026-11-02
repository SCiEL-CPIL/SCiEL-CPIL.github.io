import time
import numpy as np

DIMENSIONS = [3, 3, 4]
_SEARCH_TIMEOUT = 110.0 # 10s margin

def _build_target_tensor(n, m, p):
    T = np.ones((n, m, p), dtype=float)
    return T

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Reshape to (n, m, R), (m, p, R), (p, n, R)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # W needs to be permuted to (n, p, R) to match i, k indices
    # W3[k, i, r] -> W_perm[i, k, r]
    W_perm = np.transpose(W3, (1, 0, 2))
    
    # Compute T
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W_perm[i,k,r]
    # We can use einsum: 'ijr,jkr,ikr->ijk'
    try:
        T_calc = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W_perm, optimize=True)
    except Exception:
        return False
    
    if not np.allclose(T_calc, np.ones((n, m, p)), atol=tol):
        return False
        
    # Check half-integer constraint
    def is_half_int(arr):
        return np.allclose(arr * 2, np.round(arr * 2), atol=1e-9)
        
    return is_half_int(U) and is_half_int(V) and is_half_int(W)

def _residual(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    W_perm = np.transpose(W3, (1, 0, 2))
    T_calc = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W_perm, optimize=True)
    return np.linalg.norm(T_calc - 1.0)

def _snap_to_half_int(arr):
    return np.round(arr * 2.0) / 2.0

def _try_rank(n, m, p, R, deadline):
    # Initialize with random small values
    # Use a structured init: some entries 1, others 0
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Ensure non-trivial start
    np.random.seed(42) # Deterministic for reproducibility in this attempt? No, random is better for search.
    
    # Strategy: Random sparse init
    density = 0.3
    for r in range(R):
        # Pick some indices to be non-zero
        idx_u = np.random.choice(n*m, size=int(n*m*density), replace=False)
        idx_v = np.random.choice(m*p, size=int(m*p*density), replace=False)
        idx_w = np.random.choice(p*n, size=int(p*n*density), replace=False)
        
        U[idx_u, r] = np.random.choice([-1, 1])
        V[idx_v, r] = np.random.choice([-1, 1])
        W[idx_w, r] = np.random.choice([-1, 1])
        
    # Gradient Descent
    lr = 0.01
    best_res = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        res = _residual(U, V, W, n, m, p)
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if res < 1e-8:
            U_snapped = _snap_to_half_int(U)
            V_snapped = _snap_to_half_int(V)
            W_snapped = _snap_to_half_int(W)
            if _verify_exact(U_snapped, V_snapped, W_snapped, n, m, p):
                return U_snapped, V_snapped, W_snapped
                
        # Compute gradients numerically (approx) or use finite difference
        # For speed, we will just do random perturbations (Simulated Annealing style)
        # This is more robust than GD for discrete targets
        
        # Pick a random coordinate
        fc = np.random.randint(0, 3)
        if fc == 0:
            mat = U; nr = n*m
        elif fc == 1:
            mat = V; nr = m*p
        else:
            mat = W; nr = p*n
            
        row = np.random.randint(0, nr)
        col = np.random.randint(0, R)
        
        # Try steps: -1, -0.5, 0.5, 1
        old_val = mat[row, col]
        step_val = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        mat[row, col] = old_val + step_val
        
        new_res = _residual(U, V, W, n, m, p)
        
        # Accept if better, or with small probability if worse (annealing)
        if new_res < res or np.random.rand() < 0.05:
            res = new_res
        else:
            mat[row, col] = old_val
            
    # Final check
    U_snapped = _snap_to_half_int(best_U)
    V_snapped = _snap_to_half_int(best_V)
    W_snapped = _snap_to_half_int(best_W)
    if _verify_exact(U_snapped, V_snapped, W_snapped, n, m, p):
        return U_snapped, V_snapped, W_snapped
        
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    t0 = time.time()
    hard_deadline = t0 + _SEARCH_TIMEOUT
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up
    # Lower bound is 12.
    # We know 36 works.
    # Try 12 to 35.
    # To save time, we can try a few promising ranks: 12, 15, 20, 23, 30
    # But a linear scan with time slicing is safer.
    
    best_result = None
    
    # We will allocate time proportionally.
    # Total time ~ 110s.
    # Ranks: 12, 13, ..., 35 (24 ranks).
    # ~4.5s per rank.
    
    for R in range(12, 36):
        if time.time() > hard_deadline:
            break
            
        # Give each rank a time slice
        rank_deadline = time.time() + 4.0
        
        # Multiple restarts within the time slice
        attempts = 0
        while time.time() < rank_deadline:
            result = _try_rank(n, m, p, R, rank_deadline)
            if result is not None:
                best_result = result
                break # Found one for this rank, move to next (lower is better, so we stop at first found for this R? No, we want minimal R. So if we find R, we can stop completely? Yes, because we iterate R from low to high.)
            attempts += 1
            if attempts > 5: # Limit restarts per rank to avoid getting stuck
                break
                
        if best_result is not None:
            # We found a solution for rank R. Since we started from 12, this is the minimal found so far.
            # We can return immediately?
            # Yes, the goal is rank-minimal.
            # However, maybe we can find a lower one? No, we started at 12.
            # So if we find R=15, we haven't checked 12-14?
            # We should check 12, 13, 14 first.
            # If we find a solution at R, it is the minimal possible if we haven't found one lower.
            # So we can return.
            pass

    if best_result is None:
        U, V, W = _schoolbook()
    else:
        U, V, W = best_result
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
