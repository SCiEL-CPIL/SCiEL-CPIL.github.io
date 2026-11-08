"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact within tolerance"""
    val = _compute_val(U, V, W, n, m, p)
    return np.max(np.abs(val - 1.0)) < tol

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-rank decomposition using structured initialization and optimization."""
    np.random.seed(42 + rank)
    
    # Try multiple initializations
    for trial in range(5):
        if time.time() > deadline:
            return None
        
        # Different initialization strategies
        if trial == 0:
            # Pure random
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
        elif trial == 1:
            # Structured: group by k
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
            U += np.random.randn(n * m, rank) * 0.1
            V += np.random.randn(m * p, rank) * 0.1
            W += np.random.randn(p * n, rank) * 0.1
        elif trial == 2:
            # Block structure: exploit the 4x9 aspect ratio
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            # Assign columns in blocks
            cols_per_block = rank // p
            for k in range(p):
                start = k * cols_per_block
                end = start + cols_per_block
                for c in range(start, min(end, rank)):
                    i = (c - start) % n
                    j = ((c - start) // n) % m
                    if i * m + j < n * m:
                        U[i * m + j, c] = 1.0
                    if j * p + k < m * p:
                        V[j * p + k, c] = 1.0
                    if k * n + i < p * n:
                        W[k * n + i, c] = 1.0
            U += np.random.randn(n * m, rank) * 0.05
            V += np.random.randn(m * p, rank) * 0.05
            W += np.random.randn(p * n, rank) * 0.05
        elif trial == 3:
            # Sparse random
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
            # Zero out some entries
            mask = np.random.rand(n * m, rank) < 0.3
            U[mask] = 0
            mask = np.random.rand(m * p, rank) < 0.3
            V[mask] = 0
            mask = np.random.rand(p * n, rank) < 0.3
            W[mask] = 0
        else:
            # Cyclic structure
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for c in range(rank):
                i = (c * 7) % n
                j = (c * 13) % m
                k = (c * 19) % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            U += np.random.randn(n * m, rank) * 0.2
            V += np.random.randn(m * p, rank) * 0.2
            W += np.random.randn(p * n, rank) * 0.2
        
        # Check if already valid
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return (U, V, W)
        
        # Adam optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        max_steps = 150
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Compute gradients
            grad_U, grad_V, grad_W = _compute_gradients(U, V, W, n, m, p)
            
            # Update Adam moments
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
            
            # Bias-corrected moments
            t = step + 1
            m_U_hat = m_U / (1 - beta1 ** t)
            v_U_hat = v_U / (1 - beta2 ** t)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            
            # Update parameters
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Track best
            if step % 5 == 0:
                val = _compute_val(U, V, W, n, m, p)
                loss = np.max(np.abs(val - 1.0))
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-6:
                    return (U, V, W)
        
        # Check best found
        if best_loss < 1e-6:
            return (best_U, best_V, best_W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit, keep 5s safety margin
    
    # Generate schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 36
    # Try strategic ranks
    candidate_ranks = list(range(120, 35, -1))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        # Try to find a decomposition of this rank
        result = _try_rank(n, m, p, target_rank, deadline - 10)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
