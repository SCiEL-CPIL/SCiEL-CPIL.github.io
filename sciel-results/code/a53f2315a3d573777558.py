import time
import numpy as np
import sys

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 110  # seconds, leave 10s safety margin

def _create_target_tensor(n, m, p):
    """
    Create the target tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i with consistent indices, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _compute_error_and_grad(U, V, W, T):
    """
    Compute the squared Frobenius error and gradients for U, V, W.
    U: (nm, R), V: (mp, R), W: (pn, R)
    T: (nm, mp, pn)
    """
    R = U.shape[1]
    nm, mp, pn = T.shape
    
    # Compute predicted tensor T_hat[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    T_hat = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    Residual = T - T_hat
    Error = np.sum(Residual ** 2)
    
    # Gradients
    grad_U = np.einsum('abc,br,cr->ar', Residual, V, W, optimize=True) * (-2.0)
    grad_V = np.einsum('abc,ar,cr->br', Residual, U, W, optimize=True) * (-2.0)
    grad_W = np.einsum('abc,ar,br->cr', Residual, U, V, optimize=True) * (-2.0)
    
    return Error, grad_U, grad_V, grad_W

def _optimize_decomposition(T, rank, max_iters=1000, lr=0.1, tol=1e-10, seed=0):
    """
    Optimize U, V, W for a given rank using Adam optimizer with cosine decay and clipping.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = T.shape
    
    # Initialize with random values
    # Scale is important: too small -> slow, too large -> unstable
    scale = 0.5
    U = rng.randn(nm, rank) * scale
    V = rng.randn(mp, rank) * scale
    W = rng.randn(pn, rank) * scale
    
    # Adam state
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Cosine decay parameters
    warmup_iters = 50
    total_iters = max_iters
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            return best_err, best_U, best_V, best_W
        
        err, gU, gV, gW = _compute_error_and_grad(U, V, W, T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < tol:
            break
            
        t += 1
        
        # Learning rate schedule: Warmup then Cosine Decay
        if t < warmup_iters:
            lr_t = lr * (t / warmup_iters)
        else:
            progress = (t - warmup_iters) / (total_iters - warmup_iters)
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Gradient Clipping
        max_norm = 1.0
        gU_norm = np.linalg.norm(gU)
        gV_norm = np.linalg.norm(gV)
        gW_norm = np.linalg.norm(gW)
        if gU_norm > max_norm: gU = gU * (max_norm / gU_norm)
        if gV_norm > max_norm: gV = gV * (max_norm / gV_norm)
        if gW_norm > max_norm: gW = gW * (max_norm / gW_norm)

        # Update U
        m_U = b1 * m_U + (1 - b1) * gU
        v_U = b2 * v_U + (1 - b2) * (gU ** 2)
        m_hat = m_U / (1 - b1**t)
        v_hat = v_U / (1 - b2**t)
        U -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        
        # Update V
        m_V = b1 * m_V + (1 - b1) * gV
        v_V = b2 * v_V + (1 - b2) * (gV ** 2)
        m_hat = m_V / (1 - b1**t)
        v_hat = v_V / (1 - b2**t)
        V -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        
        # Update W
        m_W = b1 * m_W + (1 - b1) * gW
        v_W = b2 * v_W + (1 - b2) * (gW ** 2)
        m_hat = m_W / (1 - b1**t)
        v_hat = v_W / (1 - b2**t)
        W -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        
    return best_err, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    T = _create_target_tensor(n, m, p)
    schoolbook_rank = n * m * p  # 100
    
    # Fallback schoolbook decomposition
    U_fb = np.zeros((n * m, schoolbook_rank))
    V_fb = np.zeros((m * p, schoolbook_rank))
    W_fb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_fb[a, col] = 1.0
                V_fb[b, col] = 1.0
                W_fb[c, col] = 1.0
                col += 1
    
    # Search for lower rank
    # We prioritize ranks that are likely to be achievable.
    # Known bounds suggest rank can be significantly lower than 100.
    # We try a range of ranks, spending more time on "middle" ranks.
    
    ranks_to_try = [60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    # If time is very tight, only try a few ranks
    if time.time() - START_TIME > 10:
        ranks_to_try = [95, 90, 85, 80]
        
    for rank in ranks_to_try:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Number of restarts depends on time remaining
        time_remaining = TIME_LIMIT - time.time()
        if time_remaining < 5:
            break
            
        # Allocate time per rank
        # Each optimization run takes ~0.1-0.5s depending on convergence
        num_restarts = max(3, min(15, int(time_remaining / 3)))
        
        found = False
        for trial in range(num_restarts):
            if time.time() - START_TIME > TIME_LIMIT - 2:
                break
                
            seed = trial * 12345 + rank
            err, U, V, W = _optimize_decomposition(T, rank, max_iters=500, lr=0.1, tol=1e-10, seed=seed)
            
            if err < 1e-7:
                # Verify strictly
                T_hat = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                if np.allclose(T, T_hat, atol=1e-6, rtol=1e-6):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
    # Return schoolbook fallback
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }

# EVOLVE-BLOCK-END
