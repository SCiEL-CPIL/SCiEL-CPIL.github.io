import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 110.0

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Try ranks from 37 upwards. 
    # If we find a lower rank, we return it.
    # We prioritize lower ranks.
    target_ranks = [37, 38, 39, 40, 41, 42, 43, 44]
    
    for target_rank in target_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
            
        # More restarts for lower ranks as they are harder
        max_restarts = 15 if target_rank <= 38 else 8
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            try:
                result = search_rank_momentum(n, m, p, target_rank, deadline)
                if result is not None:
                    return result
            except Exception:
                pass

    return best_result

def search_rank_momentum(n, m, p, rank, deadline):
    """
    Uses vectorized Gradient Descent with momentum and adaptive step size.
    """
    # Multiple initialization strategies
    init_strategies = [
        'small_random',
        'structured',
        'sparse'
    ]
    
    for init_strategy in init_strategies:
        if time.time() > deadline - 10:
            break
            
        if init_strategy == 'small_random':
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
        elif init_strategy == 'structured':
            # Initialize with some structure based on known patterns
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for r in range(rank):
                # Try to build up the decomposition gradually
                if r < n:
                    for j in range(m):
                        U[j * m + 0, r] = 1.0
                elif r < n * m:
                    idx = r - n
                    i = idx // m
                    j = idx % m
                    U[i * m + j, r] = 1.0
                    V[j * p + 0, r] = 1.0
                else:
                    idx = r - n * m
                    k = idx % p
                    V[0 * p + k, r] = 1.0
                    W[k * n + 0, r] = 1.0
        elif init_strategy == 'sparse':
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for r in range(rank):
                # Random sparse initialization
                u_idx = np.random.randint(0, n * m, size=max(1, rank // 10))
                v_idx = np.random.randint(0, m * p, size=max(1, rank // 10))
                w_idx = np.random.randint(0, p * n, size=max(1, rank // 10))
                U[u_idx, r] = np.random.choice([-1, 1])
                V[v_idx, r] = np.random.choice([-1, 1])
                W[w_idx, r] = np.random.choice([-1, 1])
        
        result = _gradient_descent_momentum(U, V, W, n, m, p, rank, deadline)
        if result is not None:
            return result
    
    return None

def _gradient_descent_momentum(U, V, W, n, m, p, rank, deadline):
    """
    Gradient descent with momentum and adaptive step size.
    """
    U3 = U.copy().reshape(n, m, rank)
    V3 = V.copy().reshape(m, p, rank)
    W3 = W.copy().reshape(p, n, rank)
    
    # Momentum buffers
    vU3 = np.zeros_like(U3)
    vV3 = np.zeros_like(V3)
    vW3 = np.zeros_like(W3)
    
    momentum = 0.9
    step_size = 0.01
    best_error = np.inf
    best_U3, best_V3, best_W3 = U3.copy(), V3.copy(), W3.copy()
    
    max_iters = 5000
    patience = 500
    no_improve_count = 0
    
    for iter in range(max_iters):
        if time.time() > deadline - 5:
            break
            
        # Compute T
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.sum((T - 1.0)**2)
        
        if error < 1e-10:
            return _try_snap(U3, V3, W3, n, m, p)
        
        if error < best_error:
            best_error = error
            best_U3, best_V3, best_W3 = U3.copy(), V3.copy(), W3.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > patience and iter > 100:
            # Reset to best and reduce step size
            U3, V3, W3 = best_U3.copy(), best_V3.copy(), best_W3.copy()
            vU3, vV3, vW3 = np.zeros_like(U3), np.zeros_like(V3), np.zeros_like(W3)
            step_size *= 0.5
            no_improve_count = 0
        
        # Compute gradients
        dU3 = 2.0 * np.einsum('ijk,jkr,kir->ijr', T - 1.0, V3, W3)
        dV3 = 2.0 * np.einsum('ijk,ijr,kir->jkr', T - 1.0, U3, W3)
        dW3 = 2.0 * np.einsum('ijk,ijr,jkr->kir', T - 1.0, U3, V3)
        
        # Momentum update
        vU3 = momentum * vU3 - step_size * dU3
        vV3 = momentum * vV3 - step_size * dV3
        vW3 = momentum * vW3 - step_size * dW3
        
        U3 = U3 + vU3
        V3 = V3 + vV3
        W3 = W3 + vW3
        
        # Adaptive step size: decay slowly
        if iter % 100 == 0 and iter > 0:
            step_size *= 0.98
    
    # Final check with best found
    U_flat = best_U3.reshape(-1)
    V_flat = best_V3.reshape(-1)
    W_flat = best_W3.reshape(-1)
    
    result = _try_snap(U_flat, V_flat, W_flat, n, m, p)
    if result is not None:
        return result
    
    # Try polishing with the current best
    if best_error < 1e-6:
        return _polish(U_flat, V_flat, W_flat, n, m, p, deadline)
    
    return None

def _try_snap(U, V, W, n, m, p):
    """Try to snap to half-integers and verify."""
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    error = compute_error(U_snap, V_snap, W_snap, n, m, p)
    if error < 1e-8:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try integer snap
    U_snap_int = np.round(U)
    V_snap_int = np.round(V)
    W_snap_int = np.round(W)
    error_int = compute_error(U_snap_int, V_snap_int, W_snap_int, n, m, p)
    if error_int < 1e-8:
        return {"U": U_snap_int.tolist(), "V": V_snap_int.tolist(), "W": W_snap_int.tolist()}
    
    return None

def _polish(U, V, W, n, m, p, deadline):
    """High-precision polishing pass."""
    U3 = U.copy().reshape(n, m, -1)
    V3 = V.copy().reshape(m, p, -1)
    W3 = W.copy().reshape(p, n, -1)
    rank = U3.shape[2]
    
    step_size = 0.001
    vU3 = np.zeros_like(U3)
    vV3 = np.zeros_like(V3)
    vW3 = np.zeros_like(W3)
    momentum = 0.9
    
    for iter in range(2000):
        if time.time() > deadline - 3:
            break
            
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.sum((T - 1.0)**2)
        
        if error < 1e-12:
            return _try_snap(U3.reshape(-1), V3.reshape(-1), W3.reshape(-1), n, m, p)
        
        dU3 = 2.0 * np.einsum('ijk,jkr,kir->ijr', T - 1.0, V3, W3)
        dV3 = 2.0 * np.einsum('ijk,ijr,kir->jkr', T - 1.0, U3, W3)
        dW3 = 2.0 * np.einsum('ijk,ijr,jkr->kir', T - 1.0, U3, V3)
        
        vU3 = momentum * vU3 - step_size * dU3
        vV3 = momentum * vV3 - step_size * dV3
        vW3 = momentum * vW3 - step_size * dW3
        
        U3 = U3 + vU3
        V3 = V3 + vV3
        W3 = W3 + vW3
        
        if iter % 200 == 0:
            step_size *= 0.95
    
    return _try_snap(U3.reshape(-1), V3.reshape(-1), W3.reshape(-1), n, m, p)

def compute_error(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.sum((T - 1.0)**2)
    return error
