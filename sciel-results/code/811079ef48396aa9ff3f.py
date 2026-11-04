import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid using vectorized computation."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute all tensor entries: T[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
    T = np.einsum('uc,vc,wc->uvw', U, V, W)
    
    # Check the 1s: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                if abs(T[u,v,w] - 1.0) > tol:
                    return False
    
    # Check the 0s: all other (u,v,w) combinations should be 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                if (u, v, w) not in valid:
                    if abs(T[u,v,w]) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _build_valid_mask(n, m, p):
    """Precompute the valid (u,v,w) mask."""
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                mask[u, v, w] = True
    return mask

def _try_optimize(n, m, p, rank, seed, max_iter=3000, tol=1e-12):
    """Try to find a valid decomposition of given rank using proper optimization."""
    rng = np.random.RandomState(seed)
    
    # Build valid mask
    valid_mask = _build_valid_mask(n, m, p)
    zero_mask = ~valid_mask
    
    # Initialize with random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Target: 1 for valid entries, 0 for others
    # We'll work with the full tensor directly
    
    for it in range(max_iter):
        # Compute full tensor
        T = np.einsum('uc,vc,wc->uvw', U, V, W)
        
        # Compute error: sum of (T - target)^2 where target is 1 for valid, 0 for invalid
        # Equivalent to: sum(T[zero_mask]^2) + sum((T[valid_mask] - 1)^2)
        err = np.sum(T[zero_mask] ** 2) + np.sum((T[valid_mask] - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        # Compute gradients
        # For valid entries: gradient w.r.t. T[u,v,w] is (T[u,v,w] - 1)
        # For invalid entries: gradient w.r.t. T[u,v,w] is T[u,v,w]
        # dT/du = V[v,c]*W[w,c], etc.
        
        # Build the full gradient tensor for T
        # grad_T[u,v,w] = 2*(T[u,v,w] - 1) if valid, 2*T[u,v,w] if invalid
        grad_T = np.where(valid_mask, 2*(T - 1.0), 2*T)
        
        # dU[u,c] = sum_{v,w} grad_T[u,v,w] * V[v,c] * W[w,c]
        dU = np.einsum('uvw,vc,wc->uc', grad_T, V, W)
        dV = np.einsum('uvw,uc,wc->vc', grad_T, U, W)
        dW = np.einsum('uvw,uc,vc->wc', grad_T, U, V)
        
        # Adaptive learning rate
        lr = 0.01 / (1 + it * 0.001)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, best_err
    return None

def _try_structured(n, m, p, rank, seed):
    """Try a structured initialization based on block patterns."""
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use a mix of random and structured initialization
    for c in range(rank):
        if c % 4 == 0:
            U[:, c] = rng.randn(n * m) * 0.3
            V[:, c] = rng.randn(m * p) * 0.3
            W[:, c] = rng.randn(p * n) * 0.3
        elif c % 4 == 1:
            # Sign-structured
            U[:, c] = rng.choice([-1, 1], size=n*m) * 0.3
            V[:, c] = rng.choice([-1, 1], size=m*p) * 0.3
            W[:, c] = rng.choice([-1, 1], size=p*n) * 0.3
        elif c % 4 == 2:
            # Sparse: only a few non-zero entries per column
            nz_u = rng.choice(n*m, size=min(3, n*m), replace=False)
            nz_v = rng.choice(m*p, size=min(3, m*p), replace=False)
            nz_w = rng.choice(p*n, size=min(3, p*n), replace=False)
            U[nz_u, c] = rng.randn(len(nz_u))
            V[nz_v, c] = rng.randn(len(nz_v))
            W[nz_w, c] = rng.randn(len(nz_w))
        else:
            # Small random
            U[:, c] = rng.randn(n * m) * 0.1
            V[:, c] = rng.randn(m * p) * 0.1
            W[:, c] = rng.randn(p * n) * 0.1
    
    return U, V, W

def _polish(U, V, W, n, m, p, max_iter=5000, tol=1e-14):
    """High-precision polish pass."""
    valid_mask = _build_valid_mask(n, m, p)
    zero_mask = ~valid_mask
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        T = np.einsum('uc,vc,wc->uvw', U, V, W)
        err = np.sum(T[zero_mask] ** 2) + np.sum((T[valid_mask] - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        grad_T = np.where(valid_mask, 2*(T - 1.0), 2*T)
        dU = np.einsum('uvw,vc,wc->uc', grad_T, V, W)
        dV = np.einsum('uvw,uc,wc->vc', grad_T, U, W)
        dW = np.einsum('uvw,uc,vc->wc', grad_T, U, V)
        
        # Use smaller learning rate for polish
        lr = 0.001 / (1 + it * 0.0005)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    min_rank = max(n*p, m*p, n*m)
    
    # Try ranks from min_rank upward
    max_rank_to_try = min(60, n * m * p)
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple seeds for each rank
        max_seeds = 6 if rank < 35 else 3
        for seed in range(max_seeds):
            if time.time() > deadline - 30:
                break
            
            # Method 1: Pure optimization with proper gradients
            result = _try_optimize(n, m, p, rank, seed, max_iter=2000)
            if result is not None:
                U_opt, V_opt, W_opt, err = result
                
                # Try to round to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
                
                # If rounding didn't work, try using the raw values
                # and see if they're close enough
                U_list_raw = U_opt.tolist()
                V_list_raw = V_opt.tolist()
                W_list_raw = W_opt.tolist()
                
                if _verify(U_list_raw, V_list_raw, W_list_raw, n, m, p, tol=1e-6):
                    # Polish with high precision
                    U_pol, V_pol, W_pol, err_pol = _polish(U_opt, V_opt, W_opt, n, m, p)
                    
                    # Try rounding the polished version
                    U_round2 = np.round(U_pol * 2) / 2
                    V_round2 = np.round(V_pol * 2) / 2
                    W_round2 = np.round(W_pol * 2) / 2
                    
                    U_list2 = U_round2.tolist()
                    V_list2 = V_round2.tolist()
                    W_list2 = W_round2.tolist()
                    
                    if _verify(U_list2, V_list2, W_list2, n, m, p):
                        best_result = {"U": U_list2, "V": V_list2, "W": W_list2}
                        best_rank = rank
                        return best_result
                    
                    # Use raw polished values if they verify
                    U_list2_raw = U_pol.tolist()
                    V_list2_raw = V_pol.tolist()
                    W_list2_raw = W_pol.tolist()
                    
                    if _verify(U_list2_raw, V_list2_raw, W_list2_raw, n, m, p, tol=1e-6):
                        best_result = {"U": U_list2_raw, "V": V_list2_raw, "W": W_list2_raw}
                        best_rank = rank
                        return best_result
            
            # Method 2: Structured initialization + optimization
            if time.time() > deadline - 30:
                break
            U_struct, V_struct, W_struct = _try_structured(n, m, p, rank, seed)
            
            # Optimize from structured init with proper gradients
            valid_mask = _build_valid_mask(n, m, p)
            zero_mask = ~valid_mask
            
            best_err = np.inf
            for it in range(1500):
                T = np.einsum('uc,vc,wc->uvw', U_struct, V_struct, W_struct)
                err = np.sum(T[zero_mask] ** 2) + np.sum((T[valid_mask] - 1.0) ** 2)
                
                if err < best_err:
                    best_err = err
                
                if err < 1e-12:
                    break
                
                grad_T = np.where(valid_mask, 2*(T - 1.0), 2*T)
                dU = np.einsum('uvw,vc,wc->uc', grad_T, V_struct, W_struct)
                dV = np.einsum('uvw,uc,wc->vc', grad_T, U_struct, W_struct)
                dW = np.einsum('uvw,uc,vc->wc', grad_T, U_struct, V_struct)
                
                lr = 0.005 / (1 + it * 0.001)
                U_struct -= lr * dU
                V_struct -= lr * dV
                W_struct -= lr * dW
            
            if best_err < 1e-8:
                U_round = np.round(U_struct * 2) / 2
                V_round = np.round(V_struct * 2) / 2
                W_round = np.round(W_struct * 2) / 2
                
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
                
                # Try raw values
                U_list_raw = U_struct.tolist()
                V_list_raw = V_struct.tolist()
                W_list_raw = W_struct.tolist()
                
                if _verify(U_list_raw, V_list_raw, W_list_raw, n, m, p, tol=1e-6):
                    # Polish
                    U_pol, V_pol, W_pol, err_pol = _polish(U_struct, V_struct, W_struct, n, m, p)
                    
                    U_round2 = np.round(U_pol * 2) / 2
                    V_round2 = np.round(V_pol * 2) / 2
                    W_round2 = np.round(W_pol * 2) / 2
                    
                    U_list2 = U_round2.tolist()
                    V_list2 = V_round2.tolist()
                    W_list2 = W_round2.tolist()
                    
                    if _verify(U_list2, V_list2, W_list2, n, m, p):
                        best_result = {"U": U_list2, "V": V_list2, "W": W_list2}
                        best_rank = rank
                        return best_result
                    
                    U_list2_raw = U_pol.tolist()
                    V_list2_raw = V_pol.tolist()
                    W_list2_raw = W_pol.tolist()
                    
                    if _verify(U_list2_raw, V_list2_raw, W_list2_raw, n, m, p, tol=1e-6):
                        best_result = {"U": U_list2_raw, "V": V_list2_raw, "W": W_list2_raw}
                        best_rank = rank
                        return best_result
    
    return best_result
