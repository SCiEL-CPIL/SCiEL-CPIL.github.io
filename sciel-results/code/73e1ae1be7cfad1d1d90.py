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

def _adam_step(params, grads, moments, beta1, beta2, eps, t):
    """Perform one Adam update step"""
    m_keys, v_keys = moments
    for param, grad in zip(params, grads):
        m_key, v_key = m_keys, v_keys
        m_key[...] = beta1 * m_key + (1 - beta1) * grad
        v_key[...] = beta2 * v_key + (1 - beta2) * grad ** 2
        m_hat = m_key / (1 - beta1 ** t)
        v_hat = v_key / (1 - beta2 ** t)
        param[...] -= lr * m_hat / (np.sqrt(v_hat) + eps)

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
    # Try ranks from a reasonable lower bound up to schoolbook
    for target_rank in range(36, schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        # Try multiple initializations: start from schoolbook and randomly remove columns
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Strategy: start from schoolbook, randomly select target_rank columns,
            # add small random noise, then optimize
            np.random.seed(42 + target_rank * 100 + attempt)
            
            # Select random columns from schoolbook
            selected_cols = np.random.choice(schoolbook_rank, size=target_rank, replace=False)
            
            U = U_sb[:, selected_cols].copy()
            V = V_sb[:, selected_cols].copy()
            W = W_sb[:, selected_cols].copy()
            
            # Add small random noise to break degeneracy
            noise_scale = 0.1
            U += np.random.randn(n * m, target_rank) * noise_scale
            V += np.random.randn(m * p, target_rank) * noise_scale
            W += np.random.randn(p * n, target_rank) * noise_scale
            
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
            
            max_steps = 300
            for step in range(max_steps):
                if time.time() > deadline - 15:
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
                
                # Check convergence every 10 steps
                if step % 10 == 0:
                    if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                        # Found a good solution - do a final polish
                        # Polish with tighter tolerance and more iterations
                        lr2 = 0.001
                        for polish_step in range(100):
                            if time.time() > deadline - 10:
                                break
                            grad_U, grad_V, grad_W = _compute_gradients(U, V, W, n, m, p)
                            U -= lr2 * grad_U
                            V -= lr2 * grad_V
                            W -= lr2 * grad_W
                            if polish_step % 10 == 0:
                                if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                                    break
                        
                        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                            U_list = [[float(x) for x in row] for row in U]
                            V_list = [[float(x) for x in row] for row in V]
                            W_list = [[float(x) for x in row] for row in W]
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = target_rank
                            break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
