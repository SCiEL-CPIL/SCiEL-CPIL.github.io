"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    
    # Check all other entries are zero
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0
    if np.max(np.abs(T_check)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _build_target_tensor(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _gradient_descent(U, V, W, T_target, n, m, p, max_iter=2000, tol=1e-10, seed=42):
    """Gradient descent with adaptive learning rate and restarts."""
    rng = np.random.RandomState(seed)
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    patience = 0
    best_patience_err = float('inf')
    
    for it in range(max_iter):
        # Compute current tensor
        T_curr = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T_curr - T_target
        err = np.sum(residual ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < best_patience_err - 1e-6:
                best_patience_err = err
                patience = 0
            else:
                patience += 1
        
        if best_err < tol:
            break
        
        if patience > 100:
            # Restart with perturbation
            U = best_U + rng.randn(*best_U.shape) * 0.1
            V = best_V + rng.randn(*best_V.shape) * 0.1
            W = best_W + rng.randn(*best_W.shape) * 0.1
            patience = 0
            lr = max(0.01, lr * 0.5)
            continue
        
        # Gradient computation
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
        
        # Scale gradients by the "magnitude" of each factor
        U_norm = np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
        V_norm = np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
        W_norm = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
        
        grad_U = grad_U / (V_norm * W_norm)
        grad_V = grad_V / (U_norm * W_norm)
        grad_W = grad_W / (U_norm * V_norm)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        lr *= 0.999
    
    return best_U, best_V, best_W, best_err

def _try_rank(n, m, p, target_rank, T_target, max_seeds=5, max_iter=1500, tol=1e-10, deadline=None):
    """Try to find a decomposition of given rank."""
    rng = np.random.RandomState(12345)
    
    for seed in range(max_seeds):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Initialize with random values, scaled appropriately
        scale = 1.0 / (target_rank ** 0.5)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
        
        # Also try initialization based on schoolbook structure
        if seed == 0:
            # Perturb schoolbook-like structure
            U0, V0, W0 = _schoolbook(n, m, p)
            # Keep only target_rank columns with some mixing
            idx = rng.choice(n * m * p, size=target_rank, replace=False)
            U = U0[:, idx] + rng.randn(n * m, target_rank) * 0.1
            V = V0[:, idx] + rng.randn(m * p, target_rank) * 0.1
            W = W0[:, idx] + rng.randn(p * n, target_rank) * 0.1
        
        U, V, W, err = _gradient_descent(U, V, W, T_target, n, m, p, 
                                          max_iter=max_iter, tol=tol, seed=seed)
        
        if err < tol:
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=tol):
                return U, V, W, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p  # 42
    
    # Build target tensor once
    T_target = _build_target_tensor(n, m, p)
    
    # Try to find better decompositions using local search
    # Start from lower ranks and go up
    for target_rank in range(30, 42):
        if time.time() > deadline - 20:
            break
        
        U, V, W, success = _try_rank(n, m, p, target_rank, T_target, 
                                       max_seeds=3, max_iter=1000, 
                                       tol=1e-10, deadline=deadline)
        
        if success:
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = target_rank
            # Continue trying lower ranks
            continue
    
    # If we found a good rank, try to polish it
    if best_rank < n * m * p and time.time() < deadline - 10:
        U_curr = np.array(best_result["U"])
        V_curr = np.array(best_result["V"])
        W_curr = np.array(best_result["W"])
        curr_rank = U_curr.shape[1]
        
        # Try to reduce rank by 1
        for target_rank in range(max(20, curr_rank - 5), curr_rank):
            if time.time() > deadline - 5:
                break
            
            U, V, W, success = _try_rank(n, m, p, target_rank, T_target,
                                           max_seeds=2, max_iter=800,
                                           tol=1e-10, deadline=deadline)
            if success:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
    
    return best_result
# EVOLVE-BLOCK-END
