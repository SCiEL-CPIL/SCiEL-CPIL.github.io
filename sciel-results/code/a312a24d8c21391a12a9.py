"""
Model-and-optimize for the 3x4x6 matrix-multiplication tensor.

Searches for a rank-minimal exact tensor decomposition using numerical
optimization of the tensor factorization equations, falling back to
schoolbook (rank n*m*p) if no lower-rank exact solution is found.
"""
import numpy as np
from fractions import Fraction
import time

n, m, p = 3, 4, 6

def _build_tensor_equations(rank, U, V, W):
    """
    For each (i, j, k), check:
    sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
    
    Returns residuals.
    """
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                residuals.append(val - 1.0)
    return residuals

def _try_rank(rank, max_iter=20000, seed=42):
    """
    Try to find a rank-r decomposition using gradient-based optimization.
    Returns (U, V, W) if found, else None.
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Use a simple gradient descent with adaptive step
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # residual = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] - 1
                    res = 0.0
                    for t in range(rank):
                        res += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    res -= 1.0
                    loss += res * res
                    
                    for t in range(rank):
                        g = 2.0 * res
                        grad_U[i * m + j, t] += g * V[j * p + k, t] * W[k * n + i, t]
                        grad_V[j * p + k, t] += g * U[i * m + j, t] * W[k * n + i, t]
                        grad_W[k * n + i, t] += g * U[i * m + j, t] * V[j * p + k, t]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
        
        # Adaptive learning rate
        norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if norm > 0:
            step = lr / norm
        else:
            step = 0
        
        U -= step * grad_U
        V -= step * grad_V
        W -= step * grad_W
        
        # Decay learning rate slowly
        if it % 1000 == 0 and it > 0:
            lr *= 0.5
    
    if best_loss < 1e-8:
        # Snap to nearest integers or half-integers if close
        U_s = np.round(best_U * 2) / 2
        V_s = np.round(best_V * 2) / 2
        W_s = np.round(best_W * 2) / 2
        
        # Verify
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(rank):
                        val += U_s[i * m + j, t] * V_s[j * p + k, t] * W_s[k * n + i, t]
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            return U_s, V_s, W_s
    
    return None

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill
    
    # Try decreasing ranks from a reasonable upper bound down to lower bounds
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But known lower bounds for matrix mult rank are higher.
    # For 3x4x6, let's try from 40 down to 24
    
    best_result = None
    for rank in range(40, 23, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, max_iter=10000, seed=rank)
        if result is not None:
            best_result = (rank, result)
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = _schoolbook()
        rank = n * m * p
    else:
        rank, (U, V, W) = best_result
    
    # Convert to lists
    U_list = [[float(U[i, t]) for t in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, t]) for t in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, t]) for t in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
