import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # The problem is the direct sum of two 3x3x4 multiplications.
    # We search for a low-rank decomposition of 3x3x4.
    n_sub, m_sub, p_sub = 3, 3, 4
    
    def check_decomposition_334(U, V, W, rank):
        """Check if U, V, W form a valid decomposition for 3x3x4."""
        # U: (9, rank), V: (12, rank), W: (12, rank)
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Vectorized:
        # For each i, j, k, compute dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
        errors = 0.0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    val = np.dot(U[i * m_sub + j, :], V[j * p_sub + k, :] * W[k * n_sub + i, :])
                    if abs(val - 1) > 1e-6:
                        return False
        return True

    def objective_and_grad_334(U, V, W):
        """Compute objective and gradients for 3x3x4."""
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        obj = 0.0
        
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err = val - 1
                    obj += err * err
                    
                    grad_U[a, :] += 2 * err * V[b, :] * W[c, :]
                    grad_V[b, :] += 2 * err * U[a, :] * W[c, :]
                    grad_W[c, :] += 2 * err * U[a, :] * V[b, :]
        
        return obj, grad_U, grad_V, grad_W

    def gradient_descent_334(U, V, W, max_steps=5000, lr_init=0.01, lr_decay=0.999, tol=1e-10):
        """Run gradient descent for 3x3x4."""
        lr = lr_init
        best_obj = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_steps):
            obj, grad_U, grad_V, grad_W = objective_and_grad_334(U, V, W)
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if obj < tol:
                break
            
            grad_norm = max(np.max(np.abs(grad_U)), np.max(np.abs(grad_V)), np.max(np.abs(grad_W)))
            if grad_norm < 1e-12:
                break
            
            scale = min(1.0, 1.0 / (grad_norm + 1e-8))
            eff_lr = lr * scale
            
            U -= eff_lr * grad_U
            V -= eff_lr * grad_V
            W -= eff_lr * grad_W
            
            lr *= lr_decay
        
        return best_U, best_V, best_W, best_obj

    def try_rank_334(target_rank, num_attempts=5, seed_base=0):
        """Try to find a valid decomposition at the given rank for 3x3x4."""
        for attempt in range(num_attempts):
            np.random.seed(seed_base + attempt * 1000 + target_rank)
            
            # Initialize near schoolbook structure
            # Schoolbook for 3x3x4 has rank 36.
            # If target_rank < 36, we can't use pure schoolbook init.
            # We use random init.
            U = np.random.randn(n_sub * m_sub, target_rank) * 0.3
            V = np.random.randn(m_sub * p_sub, target_rank) * 0.3
            W = np.random.randn(p_sub * n_sub, target_rank) * 0.3
            
            # If target_rank is close to 36, we can seed with schoolbook terms
            if target_rank >= 30:
                # Use some schoolbook terms
                num_anchor = min(target_rank, 36)
                for c in range(num_anchor):
                    i = c // (m_sub * p_sub)
                    j = (c // p_sub) % m_sub
                    k = c % p_sub
                    U[i * m_sub + j, c] = 1.0
                    V[j * p_sub + k, c] = 1.0
                    W[k * n_sub + i, c] = 1.0
                if num_anchor < target_rank:
                    for c in range(num_anchor, target_rank):
                        U[:, c] = np.random.randn(n_sub * m_sub) * 0.2
                        V[:, c] = np.random.randn(m_sub * p_sub) * 0.2
                        W[:, c] = np.random.randn(p_sub * n_sub) * 0.2
            
            U, V, W, obj = gradient_descent_334(U, V, W, max_steps=2000)
            
            if check_decomposition_334(U, V, W, target_rank):
                return U, V, W, True
        
        return None, None, None, False

    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook rank for 3x3x4 is 36. Total 72.
    # We try to find R < 36.
    best_rank_sub = 36
    best_U_sub = None
    best_V_sub = None
    best_W_sub = None

    # Fallback schoolbook for 3x3x4
    U_sb = np.zeros((n_sub * m_sub, 36))
    V_sb = np.zeros((m_sub * p_sub, 36))
    W_sb = np.zeros((p_sub * n_sub, 36))
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U_sb[i * m_sub + j, col] = 1
                V_sb[j * p_sub + k, col] = 1
                W_sb[k * n_sub + i, col] = 1
                col += 1
    best_U_sub, best_V_sub, best_W_sub = U_sb, V_sb, W_sb

    # Try ranks from 35 down to 20
    for target_rank in range(35, 19, -1):
        if time.time() > deadline:
            break
        
        # More attempts for lower ranks? Or just fixed?
        # Lower ranks are harder, but maybe we should try a few.
        num_attempts = 3
        
        U, V, W, success = try_rank_334(target_rank, num_attempts=num_attempts)
        
        if success:
            best_rank_sub = target_rank
            best_U_sub = U
            best_V_sub = V
            best_W_sub = W
            # We can continue to try lower ranks, but time is tight.
            # If we find 30, we might want to try 29.
    
    # Construct the full 3x3x8 decomposition
    # Rank is 2 * best_rank_sub
    R = best_rank_sub
    total_rank = 2 * R
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    # First half (k=0..3)
    # U is same
    U_full[:, 0:R] = best_U_sub
    # V: first 12 rows are V_sub, next 12 are 0
    V_full[0:12, 0:R] = best_V_sub
    # W: first 12 rows are W_sub, next 12 are 0
    W_full[0:12, 0:R] = best_W_sub
    
    # Second half (k=4..7)
    # U is same (columns R..2R)
    U_full[:, R:2*R] = best_U_sub
    # V: first 12 rows are 0, next 12 are V_sub
    V_full[12:24, R:2*R] = best_V_sub
    # W: first 12 rows are 0, next 12 are W_sub
    W_full[12:24, R:2*R] = best_W_sub
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
