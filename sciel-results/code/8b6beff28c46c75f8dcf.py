"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _make_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_target(n, m, p):
    """Build the target matrix multiplication tensor."""
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    return target

def _compute_tensor(U, V, W):
    """Compute the decomposed tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    rank = U.shape[1]
    T = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=float)
    # Use einsum in chunks
    chunk = max(1, min(rank, 200))
    for s in range(0, rank, chunk):
        e = min(s + chunk, rank)
        T += np.einsum('ar,br,cr->abc', U[:, s:e], V[:, s:e], W[:, s:e])
    return T

def _verify(U, V, W, target, tol=1e-6):
    """Verify decomposition matches target."""
    T = _compute_tensor(U, V, W)
    return np.max(np.abs(T - target)) < tol

def _optimize_rank(n, m, p, target, rank, max_time, rng, lr=0.01):
    """Try to find a rank-R decomposition via Adam-like optimization."""
    nm, mp, pn = n * m, m * p, p * n
    U = rng.standard_normal((nm, rank)) * 0.1
    V = rng.standard_normal((mp, rank)) * 0.1
    W = rng.standard_normal((pn, rank)) * 0.1
    
    # Adam optimizer
    U_m = np.zeros_like(U); U_v = np.zeros_like(U)
    V_m = np.zeros_like(V); V_v = np.zeros_like(V)
    W_m = np.zeros_like(W); W_v = np.zeros_like(W)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start = time.time()
    deadline = start + max_time
    
    while time.time() < deadline - 2:
        t += 1
        lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        
        # Compute current tensor and loss
        T = _compute_tensor(U, V, W)
        diff = T - target
        loss = np.sum(diff**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-8:
            break
        
        # Compute gradients
        # dL/dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        # dL/dV[b,r] = sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
        # dL/dW[c,r] = sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
        
        # Efficient computation using einsum
        # dU[a,r] = einsum('abc,br,cr->ar', diff, V, W)
        # dV[b,r] = einsum('abc,ar,cr->br', diff, U, W)
        # dW[c,r] = einsum('abc,ar,br->cr', diff, U, V)
        
        dU = np.einsum('abc,br,cr->ar', diff, V, W)
        dV = np.einsum('abc,ar,cr->br', diff, U, W)
        dW = np.einsum('abc,ar,br->cr', diff, U, V)
        
        # Adam update
        U_m = beta1 * U_m + (1 - beta1) * dU
        U_v = beta2 * U_v + (1 - beta2) * dU**2
        U_hat = U_m / (1 - beta1**t)
        V_hat = U_v / (1 - beta2**t)
        U -= lr_t * U_hat / (np.sqrt(V_hat) + eps)
        
        V_m = beta1 * V_m + (1 - beta1) * dV
        V_v = beta2 * V_v + (1 - beta2) * dV**2
        V_m_hat = V_m / (1 - beta1**t)
        V_v_hat = V_v / (1 - beta2**t)
        V -= lr_t * V_m_hat / (np.sqrt(V_v_hat) + eps)
        
        W_m = beta1 * W_m + (1 - beta1) * dW
        W_v = beta2 * W_v + (1 - beta2) * dW**2
        W_m_hat = W_m / (1 - beta1**t)
        W_v_hat = W_v / (1 - beta2**t)
        W -= lr_t * W_m_hat / (np.sqrt(W_v_hat) + eps)
    
    # Verify best solution
    if _verify(best_U, best_V, best_W, target, tol=1e-5):
        return best_U, best_V, best_W, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Build target
    target = _build_target(n, m, p)
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _make_schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    rng = np.random.default_rng(42)
    
    # Try decreasing ranks
    # For 4x5x6, try ranks from lower to schoolbook
    # Known lower bound: at least max(n*p, m, n) = max(24, 5, 4) = 24
    # Let's try from a reasonable lower bound up to schoolbook
    min_rank_to_try = 20
    max_rank_to_try = best_rank - 1
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random restarts at this rank
        num_restarts = 3
        success = False
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            remaining = deadline - time.time() - 5
            if remaining < 5:
                break
            
            U, V, W, ok = _optimize_rank(n, m, p, target, rank, remaining, rng)
            if ok:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                success = True
                break
        
        if success:
            # Found a valid decomposition at this rank, try lower
            continue
        else:
            # Didn't succeed, try next rank (higher)
            continue
    
    # Convert to lists with half-integer rounding
    def to_list(mat):
        rounded = np.round(np.array(mat) * 2) / 2
        return rounded.tolist()
    
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
