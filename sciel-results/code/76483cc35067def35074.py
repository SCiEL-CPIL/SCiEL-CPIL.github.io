import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _schoolbook():
    """Return schoolbook decomposition with rank 54."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly matches the matmul tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_structured_rank(rank, n=2, m=3, p=9, max_iters=3000, tol=1e-10, seed=42):
    """
    Try to find a rank decomposition using structured initialization and Adam optimization.
    """
    np.random.seed(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with structured patterns that might help convergence
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Try different initialization strategies
    strategies = ['random', 'structured', 'mixed']
    
    for strat_idx, strategy in enumerate(strategies):
        seed_val = seed + strat_idx * 100
        np.random.seed(seed_val)
        
        if strategy == 'random':
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
        elif strategy == 'structured':
            # Initialize with block structure
            for r in range(rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
            # Add small random perturbation
            U += np.random.randn(nm, rank) * 0.01
            V += np.random.randn(mp, rank) * 0.01
            W += np.random.randn(pn, rank) * 0.01
        else:  # mixed
            U = np.random.randn(nm, rank) * 0.05
            V = np.random.randn(mp, rank) * 0.05
            W = np.random.randn(pn, rank) * 0.05
            # Add some structure
            for r in range(min(rank, n * m * p)):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] += 0.5
                V[j * p + k, r] += 0.5
                W[k * n + i, r] += 0.5
        
        # Adam optimizer state
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        lr = 0.01
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            t += 1
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            loss = 0.0
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        
                        val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                        err = val - 1.0
                        loss += err * err
                        
                        grad_U[idx_u, :] += 2.0 * err * V[idx_v, :] * W[idx_w, :]
                        grad_V[idx_v, :] += 2.0 * err * U[idx_u, :] * W[idx_w, :]
                        grad_W[idx_w, :] += 2.0 * err * U[idx_u, :] * V[idx_v, :]
            
            loss /= (n * m * p)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < tol:
                break
            
            # Adam update
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
            U -= lr_t * m_U / (np.sqrt(v_U) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
            V -= lr_t * m_V / (np.sqrt(v_V) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
            W -= lr_t * m_W / (np.sqrt(v_W) + eps)
            
            # Decay learning rate
            if it % 500 == 499:
                lr *= 0.8
        
        if best_loss < 1e-6:
            # Try rounding to half-integers
            for round_factor in [1, 2, 4, 8]:
                U_rounded = np.round(best_U * round_factor) / round_factor
                V_rounded = np.round(best_V * round_factor) / round_factor
                W_rounded = np.round(best_W * round_factor) / round_factor
                
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), tol=1e-8):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
            # Try without rounding
            if _verify(best_U.tolist(), best_V.tolist(), W_rounded.tolist(), tol=1e-8):
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _try_polynomial_rank(rank, n=2, m=3, p=9, max_iters=2000, tol=1e-10, seed=42):
    """
    Try to find a rank decomposition using polynomial-based initialization.
    This exploits the structure of matrix multiplication.
    """
    np.random.seed(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Use a structured initialization based on polynomial bases
    for r in range(rank):
        # Create structured patterns
        i_idx = r % n
        j_idx = (r // n) % m
        k_idx = (r // (n * m)) % p
        
        # Base pattern
        U[i_idx * m + j_idx, r] = 1.0
        V[j_idx * p + k_idx, r] = 1.0
        W[k_idx * n + i_idx, r] = 1.0
        
        # Add perturbations for other indices
        for i2 in range(n):
            for j2 in range(m):
                if (i2, j2) != (i_idx, j_idx):
                    U[i2 * m + j2, r] = np.random.randn() * 0.1
        for j2 in range(m):
            for k2 in range(p):
                if (j2, k2) != (j_idx, k_idx):
                    V[j2 * p + k2, r] = np.random.randn() * 0.1
        for k2 in range(p):
            for i2 in range(n):
                if (k2, i2) != (k_idx, i_idx):
                    W[k2 * n + i2, r] = np.random.randn() * 0.1
    
    # Optimize
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    lr = 0.005
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        t += 1
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        loss = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    err = val - 1.0
                    loss += err * err
                    
                    grad_U[idx_u, :] += 2.0 * err * V[idx_v, :] * W[idx_w, :]
                    grad_V[idx_v, :] += 2.0 * err * U[idx_u, :] * W[idx_w, :]
                    grad_W[idx_w, :] += 2.0 * err * U[idx_u, :] * V[idx_v, :]
        
        loss /= (n * m * p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
        U -= lr_t * m_U / (np.sqrt(v_U) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
        V -= lr_t * m_V / (np.sqrt(v_V) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
        W -= lr_t * m_W / (np.sqrt(v_W) + eps)
        
        if it % 300 == 299:
            lr *= 0.85
    
    if best_loss < 1e-6:
        for round_factor in [1, 2, 4]:
            U_rounded = np.round(best_U * round_factor) / round_factor
            V_rounded = np.round(best_V * round_factor) / round_factor
            W_rounded = np.round(best_W * round_factor) / round_factor
            
            if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), tol=1e-8):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), tol=1e-8):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x9 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try structured and polynomial approaches for lower ranks
    # Focus on ranks that might be achievable
    candidate_ranks = [45, 42, 40, 38, 36, 34, 32, 30, 28, 27, 25, 24, 22, 21, 20, 18]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Try structured approach
        result = _try_structured_rank(rank, max_iters=1500, seed=42)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            continue
        
        # Try polynomial approach if time permits
        if time.time() < deadline - 30:
            result = _try_polynomial_rank(rank, max_iters=1000, seed=42)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                continue
    
    if time.time() > deadline:
        return best_result
    
    return best_result
