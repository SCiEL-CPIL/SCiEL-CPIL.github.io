"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_random_init(n, m, p, R, max_iters=200, lr=0.01, seed=0, deadline=None):
    """Try to find a valid decomposition of rank R using random initialization and gradient descent."""
    np.random.seed(seed)
    U = np.random.randn(n*m, R) * 0.1
    V = np.random.randn(m*p, R) * 0.1
    W = np.random.randn(p*n, R) * 0.1
    
    momentum_U = np.zeros_like(U)
    momentum_V = np.zeros_like(V)
    momentum_W = np.zeros_like(W)
    beta = 0.9
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 15:
            break
        
        # Compute error and gradients
        errors = np.zeros((n, m, p))
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i*m + j, :]
                    vk = V[j*p + k, :]
                    wk = W[k*n + i, :]
                    s_val = np.dot(ui, vk * wk)
                    errors[i, j, k] = s_val - 1.0
                    
                    factor = 2.0 * (s_val - 1.0)
                    grad_U[i*m + j, :] += factor * (vk * wk)
                    grad_V[j*p + k, :] += factor * (ui * wk)
                    grad_W[k*n + i, :] += factor * (ui * vk)
        
        err_norm = np.linalg.norm(errors)
        
        if err_norm < 1e-8:
            break
        
        # Update momentum
        momentum_U = beta * momentum_U + (1 - beta) * grad_U
        momentum_V = beta * momentum_V + (1 - beta) * grad_V
        momentum_W = beta * momentum_W + (1 - beta) * grad_W
        
        # Update parameters
        U -= lr * momentum_U
        V -= lr * momentum_V
        W -= lr * momentum_W
        
        if err_norm < best_err:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-6):
        return best_U, best_V, best_W, best_err
    
    return None, None, None, best_err

def _try_structured_init(n, m, p, R, max_iters=200, lr=0.01, seed=0, deadline=None):
    """Try to find a valid decomposition of rank R using structured initialization."""
    np.random.seed(seed)
    
    # Start with schoolbook and truncate
    schoolbook_rank = n * m * p
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    U = U_school[:, :R].copy()
    V = V_school[:, :R].copy()
    W = W_school[:, :R].copy()
    
    # Add small noise
    U += np.random.randn(*U.shape) * 0.01
    V += np.random.randn(*V.shape) * 0.01
    W += np.random.randn(*W.shape) * 0.01
    
    momentum_U = np.zeros_like(U)
    momentum_V = np.zeros_like(V)
    momentum_W = np.zeros_like(W)
    beta = 0.9
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 15:
            break
        
        # Compute error and gradients
        errors = np.zeros((n, m, p))
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i*m + j, :]
                    vk = V[j*p + k, :]
                    wk = W[k*n + i, :]
                    s_val = np.dot(ui, vk * wk)
                    errors[i, j, k] = s_val - 1.0
                    
                    factor = 2.0 * (s_val - 1.0)
                    grad_U[i*m + j, :] += factor * (vk * wk)
                    grad_V[j*p + k, :] += factor * (ui * wk)
                    grad_W[k*n + i, :] += factor * (ui * vk)
        
        err_norm = np.linalg.norm(errors)
        
        if err_norm < 1e-8:
            break
        
        # Update momentum
        momentum_U = beta * momentum_U + (1 - beta) * grad_U
        momentum_V = beta * momentum_V + (1 - beta) * grad_V
        momentum_W = beta * momentum_W + (1 - beta) * grad_W
        
        # Update parameters
        U -= lr * momentum_U
        V -= lr * momentum_V
        W -= lr * momentum_W
        
        if err_norm < best_err:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-6):
        return best_U, best_V, best_W, best_err
    
    return None, None, None, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower ranks
    # Try a set of target ranks in descending order
    target_ranks = [120, 110, 100, 90, 80, 70, 60, 50, 48, 45, 40]
    
    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try both random and structured initializations
        for init_type in ['random', 'structured']:
            if time.time() > deadline - 10:
                break
                
            for seed in range(5):
                if time.time() > deadline - 5:
                    break
                    
                if init_type == 'random':
                    U_opt, V_opt, W_opt, err = _try_random_init(
                        n, m, p, R, max_iters=150, lr=0.01, seed=seed, deadline=deadline
                    )
                else:
                    U_opt, V_opt, W_opt, err = _try_structured_init(
                        n, m, p, R, max_iters=150, lr=0.01, seed=seed, deadline=deadline
                    )
                    
                if U_opt is not None:
                    best_rank = R
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    found = True
                    print(f"Found valid decomposition with rank {R} using {init_type} init")
                    break # Move to next lower rank
            
            if found:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
