"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _ranked_decomposition(U, V, W):
    """Compute the tensor sum_c U[:,c] outer V[:,c] outer W[:,c]."""
    r = U.shape[1]
    result = np.zeros(U.shape[0:1] + V.shape[0:1] + W.shape[0:1])
    for c in range(r):
        result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    return result

def _gradient_descent_step(U, V, W, T, lr=0.01):
    """One gradient descent step to minimize ||T - sum U_c V_c W_c||^2."""
    r = U.shape[1]
    n_dim, m_dim, p_dim = T.shape
    
    # Compute current reconstruction
    recon = np.zeros_like(T)
    for c in range(r):
        recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    
    error = recon - T
    
    # Gradients: dL/dU[i,c] = sum_{j,k} error[i,j,k] * V[j,c] * W[k,c]
    # Use vectorized computation
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for c in range(r):
        # dU[:,c] = einsum('ijk,j,k->i', error, V[:,c], W[:,c])
        dU[:,c] = np.einsum('ijk,j,k->i', error, V[:,c], W[:,c])
        # dV[:,c] = einsum('ijk,i,k->j', error, U[:,c], W[:,c])
        dV[:,c] = np.einsum('ijk,i,k->j', error, U[:,c], W[:,c])
        # dW[:,c] = einsum('ijk,i,j->k', error, U[:,c], V[:,c])
        dW[:,c] = np.einsum('ijk,i,j->k', error, U[:,c], V[:,c])
    
    U -= lr * dU
    V -= lr * dV
    W -= lr * dW
    
    return U, V, W

def _try_rank(n, m, p, target_rank, max_iters=5000, lr_init=0.005, seed=42):
    """Try to find a decomposition with the given target rank using gradient descent."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    best_U, best_V, best_W = None, None, None
    best_loss = np.inf
    
    # Try multiple random initializations
    n_restarts = min(3, max(1, 200 // max(1, target_rank)))
    
    for restart in range(n_restarts):
        # Initialize with small random values
        scale = 0.5
        U = rng.randn(n*m, target_rank) * scale
        V = rng.randn(m*p, target_rank) * scale
        W = rng.randn(p*n, target_rank) * scale
        
        lr = lr_init
        loss_history = []
        
        for it in range(max_iters):
            U, V, W = _gradient_descent_step(U, V, W, T, lr)
            
            # Compute loss periodically
            if it % 50 == 0 or it == max_iters - 1:
                recon = np.zeros_like(T)
                for c in range(target_rank):
                    recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                loss = np.sum((recon - T)**2)
                loss_history.append(loss)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Adaptive learning rate
                if len(loss_history) >= 2:
                    if loss_history[-1] > loss_history[-2] * 1.1:
                        lr *= 0.5
                    elif loss < 1e-6:
                        break
                
                if lr < 1e-8:
                    break
        
        # Check if we got close enough
        if best_loss < 1e-4:
            break
    
    if best_U is not None and best_loss < 1e-4:
        return best_U, best_V, best_W, best_loss
    return None, None, None, best_loss

def _snap_to_rationals(U, V, W, tol=1e-6):
    """Try to snap values to rationals with small denominators."""
    U_s = U.copy()
    V_s = V.copy()
    W_s = W.copy()
    
    for denom in [1, 2, 3, 4, 5, 6, 7, 8]:
        U_cand = np.round(U * denom) / denom
        V_cand = np.round(V * denom) / denom
        W_cand = np.round(W * denom) / denom
        
        # Check if this is close enough
        err_U = np.max(np.abs(U_cand - U))
        err_V = np.max(np.abs(V_cand - V))
        err_W = np.max(np.abs(W_cand - W))
        
        if err_U < tol and err_V < tol and err_W < tol:
            return U_cand, V_cand, W_cand, denom
    
    return U, V, W, 1

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Build the target tensor
    T = _build_tensor(n, m, p)
    
    # Try ranks in decreasing order of ambition
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Schoolbook: 96
    # Try some promising ranks
    
    candidate_ranks = [48, 40, 36, 32]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        U, V, W, loss = _try_rank(n, m, p, target_rank, max_iters=3000, lr_init=0.003, seed=42)
        
        if U is not None and loss < 1e-4:
            # Try to snap to rationals
            U_s, V_s, W_s, denom = _snap_to_rationals(U, V, W, tol=1e-5)
            
            # Verify the snapped version
            recon = np.zeros_like(T)
            for c in range(target_rank):
                recon += np.einsum('i,j,k->ijk', U_s[:,c], V_s[:,c], W_s[:,c])
            snap_loss = np.sum((recon - T)**2)
            
            if snap_loss < 1e-3:
                # Convert to lists
                U_list = [[float(x) for x in U_s[i]] for i in range(n*m)]
                V_list = [[float(x) for x in V_s[i]] for i in range(m*p)]
                W_list = [[float(x) for x in W_s[i]] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
