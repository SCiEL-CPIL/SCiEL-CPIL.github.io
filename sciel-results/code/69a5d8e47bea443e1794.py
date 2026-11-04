"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    if T.shape != (n, m, p):
        return False
    return np.allclose(T, 1.0, atol=tol, rtol=tol)

def _to_integer_or_half(U, V, W, tol=1e-6):
    """Convert entries to integer or half-integer if possible."""
    def snap(arr):
        arr2 = 2.0 * arr
        rounded = np.round(arr2)
        if np.allclose(arr2, rounded, atol=tol):
            return (rounded / 2.0).astype(float)
        return arr
    U = snap(U)
    V = snap(V)
    W = snap(W)
    return U, V, W

def _als_decompose(r, max_iter=500, tol=1e-10):
    """Try to find a rank-r decomposition using alternating least squares."""
    np.random.seed(42)
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    for it in range(max_iter):
        # Update W: For each (k,i), solve M @ w = 1_m
        # M[j,c] = U[i,j,c] * V[j,k,c]
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve M @ w = ones(m)
                try:
                    w = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = w
                except:
                    pass
        W_3d = W.reshape(p, n, r)
        
        # Update U: For each (i,j), solve M @ u = 1_p
        # M[k,c] = V[j,k,c] * W[k,i,c]
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    u = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = u
                except:
                    pass
        U_3d = U.reshape(n, m, r)
        
        # Update V: For each (j,k), solve M @ v = 1_n
        # M[i,c] = U[i,j,c] * W[k,i,c]
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    v = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = v
                except:
                    pass
        V_3d = V.reshape(m, p, r)
        
        # Check
        if it % 10 == 0 or it == max_iter - 1:
            if _verify(U, V, W, tol=tol):
                return U, V, W, True
    
    return U, V, W, False

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = n * m * p
    
    if time.time() > deadline - 5.0:
        U, V, W = _to_integer_or_half(best_U, best_V, best_W)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try ranks from lower bound upward
    # Theoretical lower bound for tensor rank of matmul is at least max(n*m, m*p, p*n)
    # but in practice higher. Let's start from 80.
    
    for target_rank in range(80, best_rank + 1):
        if time.time() > deadline - 10.0:
            break
        
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Change seed for different trials
            old_seed = np.random.get_state()
            np.random.seed(42 + trial * 1000)
            U, V, W, success = _als_decompose(target_rank, max_iter=200)
            np.random.set_state(old_seed)
            
            if success:
                U, V, W = _to_integer_or_half(U, V, W)
                if _verify(U, V, W, tol=1e-6):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    break
        
        if best_rank <= target_rank:
            break
    
    U, V, W = _to_integer_or_half(best_U, best_V, best_W)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
