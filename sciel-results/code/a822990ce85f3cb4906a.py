import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Pre-compute target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def compute_error(U, V, W):
        # S[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((S - T) ** 2)

    def compute_gradients(U, V, W):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        E = S - T
        dU = 2.0 * np.einsum('abc,br,cr->ar', E, V, W)
        dV = 2.0 * np.einsum('abc,ar,cr->br', E, U, W)
        dW = 2.0 * np.einsum('abc,ar,br->cr', E, U, V)
        return dU, dV, dW

    def snap_to_half_integers(U, V, W):
        return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

    def is_exact(U, V, W):
        return compute_error(U, V, W) < 1e-9

    # Hardcoded Rank 17 Solution
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)
    
    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)
    
    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)
    
    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)
    
    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)
    
    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)
    
    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)
    
    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)
    
    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)
    
    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)
    
    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)
    
    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)
    
    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)
    
    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)
    
    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)
    
    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)
    
    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    best_U, best_V, best_W = U17, V17, W17
    best_rank = 17

    def try_rank(target_rank, initial_U=None, initial_V=None, initial_W=None, seed=0):
        nonlocal best_U, best_V, best_W, best_rank
        if time.time() > deadline:
            return
        
        np.random.seed(seed)
        
        if initial_U is not None:
            U = initial_U.copy()
            V = initial_V.copy()
            W = initial_W.copy()
        else:
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
        lr = 0.01
        prev_err = float('inf')
        
        for step in range(2000):
            if time.time() > deadline:
                return
                
            err = compute_error(U, V, W)
            if err < 1e-9:
                U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                if is_exact(U_s, V_s, W_s):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_s, V_s, W_s
                    return
                    
            dU, dV, dW = compute_gradients(U, V, W)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if step % 50 == 0:
                U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                if is_exact(U_s, V_s, W_s):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_s, V_s, W_s
                    return

    # Try to find Rank 16
    # Strategy 1: Drop a column from Rank 17 and refine
    for drop_idx in range(17):
        if time.time() > deadline:
            break
        U_init = np.delete(U17, drop_idx, axis=1)
        V_init = np.delete(V17, drop_idx, axis=1)
        W_init = np.delete(W17, drop_idx, axis=1)
        try_rank(16, U_init, V_init, W_init, seed=drop_idx)
        if best_rank == 16:
            break

    # Strategy 2: Random Rank 16
    if best_rank > 16:
        for seed in range(50):
            if time.time() > deadline:
                break
            try_rank(16, seed=seed)
            if best_rank == 16:
                break

    # Try to find Rank 15
    if best_rank > 15:
        for seed in range(50):
            if time.time() > deadline:
                break
            try_rank(15, seed=seed)
            if best_rank == 15:
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
