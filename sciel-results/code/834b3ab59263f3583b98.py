"""Combined program for 3x3 x 3x4 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n=3, m=3, p=4):
    """Verify rank-r decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
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

def _compute_loss_and_grads_vectorized(U, V, W, n, m, p):
    """Compute loss and analytical gradients using vectorized NumPy operations."""
    rank = U.shape[1]
    
    # Build index arrays for vectorized computation
    # For each (i,j,k), we need U[i*m+j], V[j*p+k], W[k*n+i]
    # Let's precompute the products
    
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # Precompute for each (i,j,k) the product vectors
    # U_idx[i,j] = i*m+j, V_idx[j,k] = j*p+k, W_idx[k,i] = k*n+i
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]  # shape (rank,)
                vj = V[j * p + k]  # shape (rank,)
                wk = W[k * n + i]  # shape (rank,)
                
                prod = ui * vj * wk
                s = np.dot(prod, np.ones(rank))  # or np.sum(prod)
                diff = s - 1.0
                loss += diff * diff
                
                # Gradients
                grad_U[i * m + j] += 2 * diff * vj * wk
                grad_V[j * p + k] += 2 * diff * ui * wk
                grad_W[k * n + i] += 2 * diff * ui * vj
    
    return loss, grad_U, grad_V, grad_W

def _flatten_params(U, V, W):
    """Flatten U, V, W into a single vector."""
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten_params(params, n, m, p, rank):
    """Unflatten a parameter vector back into U, V, W."""
    size_U = n * m * rank
    size_V = m * p * rank
    U = params[:size_U].reshape(n * m, rank)
    V = params[size_U:size_U + size_V].reshape(m * p, rank)
    W = params[size_U + size_V:].reshape(p * n, rank)
    return U, V, W

def _loss_fn(params, n, m, p, rank):
    """Loss function for scipy optimizer."""
    U, V, W = _unflatten_params(params, n, m, p, rank)
    loss, _, _, _ = _compute_loss_and_grads_vectorized(U, V, W, n, m, p)
    return loss

def _grad_fn(params, n, m, p, rank):
    """Gradient function for scipy optimizer."""
    U, V, W = _unflatten_params(params, n, m, p, rank)
    _, gU, gV, gW = _compute_loss_and_grads_vectorized(U, V, W, n, m, p)
    return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

def _optimize_with_scipy(U_init, V_init, W_init, n, m, p, rank, max_iter=1000, timeout=20.0):
    """Optimize using scipy's L-BFGS-B."""
    start = time.time()
    params0 = _flatten_params(U_init, V_init, W_init)
    
    def loss_and_grad(params):
        U, V, W = _unflatten_params(params, n, m, p, rank)
        loss, gU, gV, gW = _compute_loss_and_grads_vectorized(U, V, W, n, m, p)
        grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        return loss, grad
    
    result = minimize(loss_and_grad, params0, jac=True, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'maxfun': 2000})
    
    if result.success or result.fun < 1e-10:
        U, V, W = _unflatten_params(result.x, n, m, p, rank)
        loss, _, _, _ = _compute_loss_and_grads_vectorized(U, V, W, n, m, p)
        if loss < 1e-8:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist(),
                "loss": loss
            }
    
    return None

def _try_scipy_optimization(n, m, p, rank, U_init, V_init, W_init, timeout=15.0):
    """Try scipy optimization with given initializations."""
    start = time.time()
    
    result = _optimize_with_scipy(U_init, V_init, W_init, n, m, p, rank, 
                                   max_iter=500, timeout=timeout)
    if result is not None:
        return result
    
    return None

def _structured_init(n, m, p, rank, seed=0):
    """Create structured initializations based on block patterns."""
    rng = np.random.RandomState(seed)
    
    # Start with a perturbed schoolbook-like pattern
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each (i,j,k) triple, assign to a random column with some probability
    # This creates a sparse structured pattern
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Assign to 1-3 random columns
                num_cols = rng.randint(1, 4)
                cols = rng.choice(rank, size=num_cols, replace=False)
                for c in cols:
                    U[i * m + j, c] += rng.uniform(0.5, 1.5)
                    V[j * p + k, c] += rng.uniform(0.5, 1.5)
                    W[k * n + i, c] += rng.uniform(0.5, 1.5)
    
    # Add some random noise
    U += rng.uniform(-0.1, 0.1, U.shape)
    V += rng.uniform(-0.1, 0.1, V.shape)
    W += rng.uniform(-0.1, 0.1, W.shape)
    
    return U, V, W

def _try_rank_decomposition(n, m, p, rank, timeout=30.0, num_attempts=5):
    """Try to find a rank-r decomposition using structured inits and scipy."""
    start = time.time()
    
    for attempt in range(num_attempts):
        if time.time() - start > timeout:
            break
        
        # Use structured initialization
        U, V, W = _structured_init(n, m, p, rank, seed=attempt * 1000 + 42)
        
        result = _try_scipy_optimization(n, m, p, rank, U, V, W, 
                                          timeout=min(10.0, timeout - (time.time() - start)))
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find lower-rank decompositions using scipy
    # Search ranks from best_rank-1 down to a reasonable lower bound
    # Theoretical lower bound for 3x3*3x4 is around 27-30
    
    for rank in range(best_rank - 1, max(best_rank - 8, 25), -1):
        if time.time() > deadline - 15:
            break
        
        # Try scipy-based optimization with structured inits
        result = _try_rank_decomposition(n, m, p, rank, 
                                          timeout=min(20.0, deadline - time.time() - 10),
                                          num_attempts=3)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = rank
                break
    
    # If we found an improvement, try to push further
    if best_rank < n * m * p:
        for rank in range(best_rank - 1, max(best_rank - 4, 22), -1):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_decomposition(n, m, p, rank,
                                              timeout=min(15.0, deadline - time.time() - 5),
                                              num_attempts=2)
            if result is not None:
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = rank
                    break
    
    return best
