"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    # Check off-diagonal entries are zero
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0
    if np.max(np.abs(T_check)) > 1e-9:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Build target tensor once
    T_target = np.zeros((n*m, m*p, p*n))
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                target_indices.append((a, b, c))
    
    # Precompute off-diagonal mask for efficient error computation
    # All entries are either diagonal (should be 1) or off-diagonal (should be 0)
    # Total entries: n*m * m*p * p*n = 6*21*14 = 1764
    # Diagonal entries: n*m*p = 42
    # Off-diagonal: 1764 - 42 = 1722
    
    def _compute_full_error(U, V, W):
        """Compute the full error: sum of (T - T_target)^2 over all entries."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - T_target
        return np.sum(diff * diff)
    
    def _optimize_rank(target_rank, max_iters=500, seed=42):
        """Try to find a rank-target_rank decomposition using Adam optimizer."""
        rng = np.random.RandomState(seed)
        
        # Initialize with structure-aware random init
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        # Adam optimizer state
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        t = 0
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Compute current tensor
            T = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - T_target
            
            # Compute full error
            err = np.sum(diff * diff)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-14:
                break
            
            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Adam update
            t += 1
            lr_t = 0.01 * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U**2
            m_hat_U = m_U / (1 - beta1**t)
            v_hat_U = v_U / (1 - beta2**t)
            U -= lr_t * m_hat_U / (np.sqrt(v_hat_U) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V**2
            m_hat_V = m_V / (1 - beta1**t)
            v_hat_V = v_V / (1 - beta2**t)
            V -= lr_t * m_hat_V / (np.sqrt(v_hat_V) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W**2
            m_hat_W = m_W / (1 - beta1**t)
            v_hat_W = v_W / (1 - beta2**t)
            W -= lr_t * m_hat_W / (np.sqrt(v_hat_W) + eps)
            
            # Clip to prevent blowup
            for arr in [U, V, W]:
                max_val = np.max(np.abs(arr))
                if max_val > 1e4:
                    arr /= max_val
        
        # Final polish: try to snap to exact values if close
        if best_err < 1e-6:
            # Do a few more iterations with smaller learning rate
            for polish_it in range(100):
                if time.time() > deadline - 2:
                    break
                T = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
                diff = T - T_target
                err = np.sum(diff * diff)
                if err < 1e-14:
                    break
                grad_U = np.einsum('abc,br,cr->ar', diff, best_V, best_W)
                grad_V = np.einsum('abc,ar,cr->br', diff, best_U, best_W)
                grad_W = np.einsum('abc,ar,br->cr', diff, best_U, best_V)
                best_U -= 0.001 * grad_U
                best_V -= 0.001 * grad_V
                best_W -= 0.001 * grad_W
        
        return best_U, best_V, best_W, best_err
    
    # Try various ranks in increasing order
    for target_rank in range(4, rank_school + 1):
        if time.time() > deadline - 15:
            break
        
        # Skip ranks that are unlikely to work based on lower bounds
        # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
        # But there are better lower bounds from tensor rank theory
        if target_rank < 21:
            continue
        
        U_opt, V_opt, W_opt, err = _optimize_rank(target_rank, max_iters=400, seed=target_rank * 7)
        
        if err < 1e-10:
            if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U_opt.tolist(),
                    "V": V_opt.tolist(),
                    "W": W_opt.tolist()
                }
                break
        
        # Try with a different seed
        if time.time() > deadline - 15:
            break
        U_opt2, V_opt2, W_opt2, err2 = _optimize_rank(target_rank, max_iters=300, seed=target_rank * 7 + 13)
        if err2 < 1e-10:
            if _verify(U_opt2.tolist(), V_opt2.tolist(), W_opt2.tolist(), n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U_opt2.tolist(),
                    "V": V_opt2.tolist(),
                    "W": W_opt2.tolist()
                }
                break
    
    return best_result
# EVOLVE-BLOCK-END
