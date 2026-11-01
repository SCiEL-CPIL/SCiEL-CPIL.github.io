"""Combined program: structured block init + analytical gradient descent for 3x3 x 3x4 tensor decomposition."""
import time
import numpy as np

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

def _loss_and_grad(U, V, W, n=3, m=3, p=4):
    """
    Compute loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    and analytical gradients for U, V, W.
    
    Let S[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    Then loss = sum_{i,j,k} (S[i,j,k] - 1)^2
    
    dL/dU[a,c] where a = i*m+j:
      = 2 * sum_{k} (S[i,j,k] - 1) * V[j*p+k,c] * W[k*n+i,c]
    
    dL/dV[b,c] where b = j*p+k:
      = 2 * sum_{i} (S[i,j,k] - 1) * U[i*m+j,c] * W[k*n+i,c]
    
    dL/dW[d,c] where d = k*n+i:
      = 2 * sum_{j} (S[i,j,k] - 1) * U[i*m+j,c] * V[j*p+k,c]
    """
    rank = U.shape[1]
    # Compute S[i,j,k] for all i,j,k
    # S[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    S = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S[i, j, k] = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
    
    loss = np.sum((S - 1.0) ** 2)
    
    # Compute gradients
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                err = 2.0 * (S[i, j, k] - 1.0)
                # dL/dU[i*m+j, c] += err * V[j*p+k, c] * W[k*n+i, c]
                grad_U[i * m + j] += err * V[j * p + k] * W[k * n + i]
                # dL/dV[j*p+k, c] += err * U[i*m+j, c] * W[k*n+i, c]
                grad_V[j * p + k] += err * U[i * m + j] * W[k * n + i]
                # dL/dW[k*n+i, c] += err * U[i*m+j, c] * V[j*p+k, c]
                grad_W[k * n + i] += err * U[i * m + j] * V[j * p + k]
    
    return loss, grad_U, grad_V, grad_W

def _structured_init(n, m, p, rank, seed=42):
    """
    Initialize U, V, W with a structured pattern that partially satisfies the constraints.
    Use a block-diagonal-like structure with some randomness.
    """
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First, try to assign each (i,j,k) triple to a "primary" column
    # Then add small perturbations to allow the optimizer to adjust
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                # Assign this triple to column col with coefficient 1
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
            if col >= rank:
                break
        if col >= rank:
            break
    
    # Add small random perturbations to all entries to break symmetry
    # and allow the optimizer to find better solutions
    U += 0.01 * rng.randn(n * m, rank)
    V += 0.01 * rng.randn(m * p, rank)
    W += 0.01 * rng.randn(p * n, rank)
    
    return U, V, W

def _gradient_descent(n, m, p, rank, max_iter=5000, lr=0.05, tol=1e-12, timeout=60.0, seed=42):
    """
    Run gradient descent with analytical gradients to find a rank-r decomposition.
    """
    start_time = time.time()
    U, V, W = _structured_init(n, m, p, rank, seed=seed)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Adaptive learning rate
    current_lr = lr
    
    for iteration in range(max_iter):
        if time.time() - start_time > timeout:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Gradient step
        U -= current_lr * gU
        V -= current_lr * gV
        W -= current_lr * gW
        
        # Adaptive learning rate: reduce if loss increased
        new_loss, _, _, _ = _loss_and_grad(U, V, W, n, m, p)
        if new_loss > loss * 1.5:
            # Revert and reduce learning rate
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
            current_lr *= 0.5
            if current_lr < 1e-6:
                break
        else:
            # Slightly increase learning rate if making good progress
            if new_loss < loss * 0.9:
                current_lr = min(current_lr * 1.1, 0.1)
    
    return best_U, best_V, best_W, best_loss

def _try_rank(n, m, p, rank, timeout=45.0, num_seeds=5):
    """Try to find a valid rank-r decomposition."""
    start_time = time.time()
    
    for seed in range(num_seeds):
        if time.time() - start_time > timeout:
            break
        
        U, V, W, loss = _gradient_descent(n, m, p, rank, 
                                           max_iter=3000, lr=0.05, 
                                           tol=1e-12, timeout=timeout, seed=seed)
        
        if loss < 1e-8:
            # Verify with exact check
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find lower-rank decompositions
    # Known lower bounds suggest rank around 23-32 is possible
    # Try from best_rank-1 down to 23
    
    for rank in range(best_rank - 1, 22, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank(n, m, p, rank, timeout=30.0, num_seeds=3)
        
        if result is not None:
            best = result
            best_rank = rank
            # Don't break - try to go lower
            # But if we found a good one, we can continue searching for even lower
    
    return best
