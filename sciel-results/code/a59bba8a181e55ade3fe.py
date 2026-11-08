"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _compute_val(U, V, W, n, m, p):
    """Compute val[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
    val = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
    return val

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact within tolerance"""
    val = _compute_val(U, V, W, n, m, p)
    return np.max(np.abs(val - 1.0)) < tol

def _make_schoolbook(n, m, p):
    """Create the schoolbook decomposition as numpy arrays"""
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

def _compute_gradients(U, V, W, n, m, p):
    """Compute analytic gradients of the error function"""
    val = _compute_val(U, V, W, n, m, p)
    diff = val - 1.0
    rank = U.shape[1]
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                grad_U[i*m+j] += 2.0 * diff[i,j,k] * V[j*p+k] * W[k*n+i]
    
    for j in range(m):
        for k in range(p):
            for i in range(n):
                grad_V[j*p+k] += 2.0 * diff[i,j,k] * U[i*m+j] * W[k*n+i]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                grad_W[k*n+i] += 2.0 * diff[i,j,k] * U[i*m+j] * V[j*p+k]
    
    return grad_U, grad_V, grad_W

def _try_ladderman_block(n, m, p):
    """
    Try to construct a decomposition using Laderman's rank-7 algorithm for 2x2 matrix mult
    as a building block, extended to 4x4 * 4x9.
    
    For 4x4 * 4x4, we can use 4 blocks of 2x2, giving rank 4*7 = 28.
    For 4x4 * 4x9, we need to handle the 9 columns.
    
    Actually, let's think more carefully. The tensor is T[i,j,k] = 1 for all i,j,k
    where i in [0,3], j in [0,3], k in [0,8].
    
    We can decompose the 4x9 product as: for each block of 2 rows of the 4x4 matrix A,
    multiply by the corresponding 2x9 block of B. But that doesn't directly help.
    
    Alternative: Use the fact that 4x9 = 4x4 + 4x4 + 4x1. 
    The 4x4 part can use rank-20 (Laderman extended), and 4x1 is rank 4.
    But this gives rank 20 + 20 + 4 = 44? No, that's not how it works.
    
    Let me try a different approach: direct numerical optimization with better initialization.
    """
    return None

def _optimize_decomposition(U, V, W, n, m, p, deadline, max_steps=500, lr=0.01):
    """
    Optimize the decomposition using Adam optimizer with gradient descent.
    Returns (success, U, V, W) where success is True if verified within tolerance.
    """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    best_err = np.inf
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, n, m, p)
        
        t = step + 1
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
        
        m_U_hat = m_U / (1 - beta1 ** t)
        v_U_hat = v_U / (1 - beta2 ** t)
        m_V_hat = m_V / (1 - beta1 ** t)
        v_V_hat = v_V / (1 - beta2 ** t)
        m_W_hat = m_W / (1 - beta1 ** t)
        v_W_hat = v_W / (1 - beta2 ** t)
        
        U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
        V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
        W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        # Check error periodically
        if step % 20 == 0:
            val = _compute_val(U, V, W, n, m, p)
            err = np.max(np.abs(val - 1.0))
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-8:
                return True, U, V, W
    
    # Final check
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-8):
        return True, best_U, best_V, best_W
    
    return False, best_U, best_V, best_W

def _polish_decomposition(U, V, W, n, m, p, deadline):
    """
    High-precision polish pass using Newton's method or similar.
    """
    lr = 0.001
    for step in range(200):
        if time.time() > deadline:
            break
        
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, n, m, p)
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if step % 10 == 0:
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
                break
    
    return U, V, W

def _try_rank_with_structured_init(n, m, p, rank, deadline, seed=42):
    """
    Try to find a rank-r decomposition using structured initialization
    inspired by block decomposition ideas.
    """
    np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Structured initialization: assign columns to (i,j,k) patterns
    # but with sharing to reduce effective rank
    for c in range(rank):
        i = c % n
        j = (c // n) % m
        k = (c // (n * m)) % p
        
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    
    # Add noise to break degeneracy
    noise = 0.5
    U += np.random.randn(n * m, rank) * noise
    V += np.random.randn(m * p, rank) * noise
    W += np.random.randn(p * n, rank) * noise
    
    success, U, V, W = _optimize_decomposition(U, V, W, n, m, p, deadline, max_steps=300, lr=0.01)
    if success:
        return U, V, W
    
    # Try with different noise levels
    for noise in [0.1, 0.3, 1.0, 2.0]:
        if time.time() > deadline - 5:
            break
        
        np.random.seed(seed + 1000)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c in range(rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        
        U += np.random.randn(n * m, rank) * noise
        V += np.random.randn(m * p, rank) * noise
        W += np.random.randn(p * n, rank) * noise
        
        success, U, V, W = _optimize_decomposition(U, V, W, n, m, p, deadline, max_steps=300, lr=0.01)
        if success:
            return U, V, W
    
    return None

def _try_rank_with_random_init(n, m, p, rank, deadline, seed=42):
    """
    Try to find a rank-r decomposition using random initialization.
    """
    np.random.seed(seed)
    
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        success, U, V, W = _optimize_decomposition(U, V, W, n, m, p, deadline, max_steps=300, lr=0.01)
        if success:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit, keep 5s safety margin
    
    # Generate schoolbook as fallback
    U_sb, V_sb, W_sb = _make_schoolbook(n, m, p)
    best_result = {
        "U": [[float(x) for x in row] for row in U_sb],
        "V": [[float(x) for x in row] for row in V_sb],
        "W": [[float(x) for x in row] for row in W_sb]
    }
    best_rank = schoolbook_rank
    
    # Lower bound: max(n*m, m*p, p*n) = 36
    # For 4x4*4x9, the bilinear complexity is at most 144
    # Try ranks from a reasonable lower bound up to schoolbook
    
    # Try structured ranks first (more likely to converge)
    candidate_ranks = []
    for r in range(40, schoolbook_rank, 2):
        candidate_ranks.append(r)
    
    # Also try some specific ranks
    for r in [44, 48, 50, 52, 56, 60, 64, 72, 80, 90, 100, 120]:
        if r not in candidate_ranks and r < schoolbook_rank:
            candidate_ranks.append(r)
    
    candidate_ranks.sort()
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        # Try structured initialization
        result = _try_rank_with_structured_init(n, m, p, target_rank, deadline - 5, seed=42 + target_rank)
        if result is not None:
            U, V, W = result
            U, V, W = _polish_decomposition(U, V, W, n, m, p, deadline - 5)
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                best_rank = target_rank
                break
        
        # Try random initialization
        if time.time() > deadline - 10:
            break
        result = _try_rank_with_random_init(n, m, p, target_rank, deadline - 5, seed=1000 + target_rank)
        if result is not None:
            U, V, W = result
            U, V, W = _polish_decomposition(U, V, W, n, m, p, deadline - 5)
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                best_rank = target_rank
                break
    
    return best_result

# EVOLVE-BLOCK-END
