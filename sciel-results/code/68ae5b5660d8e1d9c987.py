"""Matrix multiplication 3x3x5 rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]
DEADLINE = 120.0

def _verify_decomposition(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Verify that U @ V^T @ W^T reconstructs the all-ones tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    
    return np.allclose(T, 1.0, atol=tol, rtol=tol)

def _make_schoolbook(n=3, m=3, p=5):
    """Create the schoolbook decomposition of rank n*m*p."""
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

def _compute_tensor(U, V, W, n=3, m=3, p=5):
    """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return T

def _loss_and_grad(U, V, W, n=3, m=3, p=5):
    """Compute loss and gradients for the decomposition."""
    rank = U.shape[1]
    T = _compute_tensor(U, V, W, n, m, p)
    loss = np.sum((T - 1.0) ** 2)
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        
        # Gradient for U
        # dL/du_ij = 2 * (T_ijk - 1) * v_jk * w_ki
        grad_u = np.zeros((n, m))
        grad_v = np.zeros((m, p))
        grad_w = np.zeros((p, n))
        
        residual = T - 1.0
        grad_u = np.einsum('ijk,jk,ki->ij', residual, v, w)
        grad_v = np.einsum('ijk,ij,ki->jk', residual, u, w)
        grad_w = np.einsum('ijk,ij,jk->ki', residual, u, v)
        
        dU[:, c] = 2 * grad_u.flatten()
        dV[:, c] = 2 * grad_v.flatten()
        dW[:, c] = 2 * grad_w.flatten()
    
    return loss, dU, dV, dW

def _try_grad_descent(n, m, p, target_rank, max_iters=5000, lr=0.01, deadline=None, init_scale=1.0):
    """Try to find a valid decomposition via gradient descent."""
    if deadline is None:
        deadline = time.time() + 30
    
    start = time.time()
    
    # Initialize with random small values
    U = np.random.randn(n * m, target_rank) * init_scale
    V = np.random.randn(m * p, target_rank) * init_scale
    W = np.random.randn(p * n, target_rank) * init_scale
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        loss, dU, dV, dW = _loss_and_grad(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm > 0:
            effective_lr = lr * 100 / (grad_norm + 1)
        else:
            effective_lr = lr
        
        U -= effective_lr * dU
        V -= effective_lr * dV
        W -= effective_lr * dW
    
    if best_loss < 1e-8:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3x5 matrix multiplication."""
    start = time.time()
    deadline = start + DEADLINE - 5.0
    
    n, m, p = DIMENSIONS
    
    schoolbook = _make_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try ranks from 5 to 20
    for target_rank in range(5, 21):
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple initializations
        found = False
        for attempt in range(10):
            if time.time() > deadline - 5.0:
                break
            
            result = _try_grad_descent(n, m, p, target_rank, max_iters=2000, lr=0.05, deadline=deadline - 5.0, init_scale=0.5)
            if result is not None:
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    return result
        
    return best_result
