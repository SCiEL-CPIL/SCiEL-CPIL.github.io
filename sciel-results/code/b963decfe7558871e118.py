import time
import numpy as np

DIMENSIONS = [3, 3, 4]
_START_TIME = time.time()
_DEADLINE = 140  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_remove_columns(U, V, W, n, m, p, max_iter=200):
    """Iteratively try to remove columns that are linearly redundant."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    rank = U.shape[1]
    improved = True
    while improved and rank > 0 and time.time() - _START_TIME < _DEADLINE - 10:
        improved = False
        for c in range(rank):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            # Try removing column c
            mask = [x for x in range(rank) if x != c]
            U_t = U[:, mask]
            V_t = V[:, mask]
            W_t = W[:, mask]
            if _verify(U_t, V_t, W_t, n, m, p, tol=1e-8):
                U, V, W = U_t, V_t, W_t
                rank -= 1
                improved = True
                break
    return U, V, W

def _random_search(n, m, p, target_rank, n_trials=5000, seed=42):
    """Try to find a decomposition of given rank via random initialization + local search."""
    rng = np.random.RandomState(seed)
    best = None
    for trial in range(n_trials):
        if time.time() - _START_TIME > _DEADLINE - 15:
            break
        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5
        
        # Check if this random guess is close
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err < 1e-6:
            best = (U, V, W)
            break
    return best

def _polish(U, V, W, n, m, p, n_iter=100):
    """Local polish: adjust entries to reduce residual."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    for it in range(n_iter):
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
        # Compute residuals for each (i,j,k)
        # Gradient descent on U, V, W
        lr = 0.01 / (it + 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    current = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    residual = current - 1.0
                    # dU[i*m+j, c] -= lr * residual * V[j*p+k, c] * W[k*n+i, c]
                    U[i*m+j, :] -= lr * residual * (V[j*p+k, :] * W[k*n+i, :])
                    # dV[j*p+k, c] -= lr * residual * U[i*m+j, c] * W[k*n+i, c]
                    V[j*p+k, :] -= lr * residual * (U[i*m+j, :] * W[k*n+i, :])
                    # dW[k*n+i, c] -= lr * residual * U[i*m+j, c] * V[j*p+k, c]
                    W[k*n+i, :] -= lr * residual * (U[i*m+j, :] * V[j*p+k, :])
        
        if _verify(U, V, W, n, m, p, tol=1e-8):
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try to remove redundant columns
    U, V, W = _try_remove_columns(U, V, W, n, m, p)
    current_rank = U.shape[1]
    
    # Try to find lower-rank decompositions
    # For 3x3x4, try ranks from current_rank-1 down to a reasonable minimum
    min_rank_to_try = max(1, current_rank - 15)
    
    for target_rank in range(current_rank - 1, min_rank_to_try - 1, -1):
        if time.time() - _START_TIME > _DEADLINE - 20:
            break
        
        # Try random search for this rank
        result = _random_search(n, m, p, target_rank, n_trials=2000)
        if result is not None:
            U, V, W = result
            current_rank = target_rank
            # Polish
            U, V, W = _polish(U, V, W, n, m, p, n_iter=50)
            if _verify(U, V, W, n, m, p, tol=1e-8):
                break
    
    # Final verification
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
