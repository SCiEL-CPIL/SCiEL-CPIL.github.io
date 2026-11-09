"""Rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 130  # leave 20s safety margin before 150s kill

def _tensor_entry(i, j, k):
    """The matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k (standard MM)."""
    return 1

def _decompose(U, V, W, n, m, p):
    """Check if U,V,W decompose the MM tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W convention: W[k*n+i, r]
    Product: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Reshape
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # For each (i,j,k): sum_r U3[i,j,r]*V3[j,k,r]*W3[k,i,r] == 1
    # Efficient: for each r, outer product
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _decompose_fast(U, V, W, n, m, p):
    """Faster decomposition check using numpy."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Compute sum over r for all (i,j,k)
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a batched contraction
    result = np.zeros((n, m, p))
    for r in range(rank):
        # U[i,j,r] * V[j,k,r] * W[k,i,r]
        u = U3[:, :, r]  # (n, m)
        v = V3[:, :, r]  # (m, p)
        w = W3[:, :, r]  # (p, n)
        # For each (i,j,k): u[i,j] * v[j,k] * w[k,i]
        # This is element-wise product with appropriate broadcasting
        # u[i,j] * v[j,k] -> (n, p) via sum over j? No, it's element-wise.
        # Actually: for fixed r, the contribution is u[i,j] * v[j,k] * w[k,i]
        # Let's compute via: (u * v * w) but with indices (i,j), (j,k), (k,i)
        # This is a 3-tensor contraction. Let me do it explicitly.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(result, 1.0, atol=1e-9)

def _decompose_vectorized(U, V, W, n, m, p):
    """Most efficient check."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: sum over r
    # U3: (i,j,r), V3: (j,k,r), W3: (k,i,r)
    # Result: (i,j,k)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def _random_init(n, m, p, rank, seed=0):
    """Random initialization with small values."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    return U, V, W

def _gradient_descent(U, V, W, n, m, p, lr=0.01, max_iter=200, tol=1e-12):
    """Gradient descent to minimize ||T - decompose(U,V,W)||^2."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        # Compute residual
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = result - 1.0
        
        # Gradients
        # d/dU3[i,j,r] = sum_{k} residual[i,j,k] * V3[j,k,r] * W3[k,i,r]
        # d/dV3[j,k,r] = sum_{i} residual[i,j,k] * U3[i,j,r] * W3[k,i,r]
        # d/dW3[k,i,r] = sum_{j} residual[i,j,k] * U3[i,j,r] * V3[j,k,r]
        
        dU = np.einsum('ijk,jkr,kir->ijr', residual, V3, W3)
        dV = np.einsum('ijk,ijr,kir->jkr', residual, U3, W3)
        dW = np.einsum('ijk,ijr,jkr->kir', residual, U3, V3)
        
        # Update
        U3 -= lr * dU
        V3 -= lr * dV
        W3 -= lr * dW
        
        # Check convergence
        err = np.max(np.abs(residual))
        if err < tol:
            break
    
    # Convert back
    U = U3.reshape(n * m, rank)
    V = V3.reshape(m * p, rank)
    W = W3.reshape(p * n, rank)
    return U, V, W

def _try_rank(n, m, p, rank, max_attempts=10, seed_base=0):
    """Try to find a decomposition of given rank."""
    for attempt in range(max_attempts):
        if time.time() - START_TIME > TIME_LIMIT:
            return None
        seed = seed_base + attempt
        U, V, W = _random_init(n, m, p, rank, seed)
        
        # Try gradient descent
        U, V, W = _gradient_descent(U, V, W, n, m, p, lr=0.05, max_iter=500)
        
        if _decompose_vectorized(U, V, W, n, m, p):
            return U, V, W
        
        # Try with different learning rates
        U, V, W = _random_init(n, m, p, rank, seed + 1000)
        U, V, W = _gradient_descent(U, V, W, n, m, p, lr=0.01, max_iter=500)
        
        if _decompose_vectorized(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    best = (U.copy(), V.copy(), W.copy())
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Known: 3x3x3 has rank 23 (Strassen-like), 3x3x4 might be lower than 36
    # Let's try ranks from low to high
    for rank in range(1, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        result = _try_rank(n, m, p, rank, max_attempts=5, seed_base=rank * 100)
        if result is not None:
            U, V, W = result
            best = (U.copy(), V.copy(), W.copy())
            best_rank = rank
            break  # Found the minimum, stop
    
    # Convert to lists as required
    U_list = best[0].tolist()
    V_list = best[1].tolist()
    W_list = best[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
