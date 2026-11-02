"""Matrix multiplication 2x3x7: search for a lower-rank exact tensor decomposition."""

import time
import numpy as np

DIMENSIONS = [2, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0
MARGIN = 10.0

def _build_tensor(n, m, p):
    """Build the 3-way matrix multiplication tensor T[i,j,k] with shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Check if U @ V @ W^T recovers the tensor exactly.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The reconstruction is T_reconstructed[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rank = U.shape[1]
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    T_true = _build_tensor(n, m, p)
    return np.allclose(T_rec, T_true, atol=1e-9)

def _schoolbook():
    """Return the schoolbook rank-42 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _random_initialization(n, m, p, rank, rng):
    """Randomly initialize U, V, W for a given rank."""
    U = rng.standard_normal((n * m, rank)) * 0.1
    V = rng.standard_normal((m * p, rank)) * 0.1
    W = rng.standard_normal((p * n, rank)) * 0.1
    return U, V, W

def _gradient_step(U, V, W, T, lr):
    """One gradient step to minimize ||T - einsum(U,V,W)||^2."""
    # Compute residual
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    R = T_rec - T  # residual to minimize
    
    # Gradients:
    # dL/dU[a,r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
    # dL/dV[b,r] = sum_{a,c} R[a,b,c] * U[a,r] * W[c,r]
    # dL/dW[c,r] = sum_{a,b} R[a,b,c] * U[a,r] * V[b,r]
    
    dU = np.einsum('abc,br,cr->ar', R, V, W)
    dV = np.einsum('abc,ar,cr->br', R, U, W)
    dW = np.einsum('abc,ar,br->cr', R, U, V)
    
    U -= lr * dU
    V -= lr * dV
    W -= lr * dW
    return U, V

def _try_rank(n, m, p, target_rank, rng, max_iters=500, lr=0.01):
    """Try to find a decomposition of given rank via gradient descent."""
    T = _build_tensor(n, m, p)
    U, V, W = _random_initialization(n, m, p, target_rank, rng)
    
    # Adaptive learning rate
    best_loss = np.inf
    patience = 100
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - MARGIN:
            break
        
        U, V, W = _gradient_step(U, V, W, T, lr)
        
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        loss = np.sum((T_rec - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            no_improve = 0
        else:
            no_improve += 1
        
        # Early exit if close to zero
        if loss < 1e-12:
            return U, V, W
        
        # Reduce learning rate if no improvement
        if no_improve > patience:
            lr *= 0.5
            no_improve = 0
            if lr < 1e-6:
                break
    
    return U, V, W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    
    # Try to find lower-rank decompositions
    # Start from lower ranks and go up
    for target_rank in range(20, 42):
        if time.time() - START_TIME > TIME_LIMIT - MARGIN:
            break
        
        found = False
        # Try multiple random restarts
        max_restarts = 3
        for attempt in range(max_restarts):
            if time.time() - START_TIME > TIME_LIMIT - MARGIN:
                break
            
            rng = np.random.default_rng(seed=target_rank * 1000 + attempt)
            U, V, W = _try_rank(n, m, p, target_rank, rng, max_iters=300, lr=0.05)
            
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            found = True  # We tried, but didn't find it
        
        if not found:
            continue
    
    # Fallback to schoolbook
    return schoolbook
