import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _validate_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast vectorized validation using einsum."""
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape and use einsum
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.max(np.abs(T - 1.0)) < tol

def _snap_to_half_integers(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    snapped = arr.copy()
    # Try snapping to multiples of 0.5
    candidates = np.round(snapped * 2) / 2
    if np.max(np.abs(snapped - candidates)) < tol:
        return candidates
    # Try snapping to integers
    candidates_int = np.round(snapped)
    if np.max(np.abs(snapped - candidates_int)) < tol:
        return candidates_int
    return snapped

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _try_structured_decomposition(n, m, p, rank, seed=0, time_budget=10.0):
    """Try structured initializations for a given rank."""
    deadline = time.time() + time_budget
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Block-structured initialization
    # Try to exploit the tensor structure with block patterns
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        if attempt % 4 == 0:
            # Kronecker-like structure
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
        elif attempt % 4 == 1:
            # Sparse initialization
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            # Zero out some entries to encourage sparsity
            for _ in range(rank // 2):
                U[np.random.randint(n * m), np.random.randint(rank)] = 0
                V[np.random.randint(m * p), np.random.randint(rank)] = 0
                W[np.random.randint(p * n), np.random.randint(rank)] = 0
        elif attempt % 4 == 2:
            # Block diagonal structure
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            block_size = max(1, rank // 4)
            for b in range(min(4, rank // block_size)):
                start = b * block_size
                end = min(start + block_size, rank)
                U[:n * m, start:end] = rng.randn(n * m, end - start) * 0.3
                V[:m * p, start:end] = rng.randn(m * p, end - start) * 0.3
                W[:p * n, start:end] = rng.randn(p * n, end - start) * 0.3
        else:
            # Geometric reparametrization: apply random linear transforms to schoolbook
            U_s, V_s, W_s = _schoolbook(n, m, p)
            # Take a subset of columns
            idx = rng.choice(n * m * p, size=min(rank, n * m * p), replace=False)
            U = U_s[:, idx]
            V = V_s[:, idx]
            W = W_s[:, idx]
            # Apply random orthogonal transforms to escape local minima
            if rank > 0:
                Q_U = np.eye(rank) + 0.1 * rng.randn(rank, rank)
                Q_V = np.eye(rank) + 0.1 * rng.randn(rank, rank)
                Q_W = np.eye(rank) + 0.1 * rng.randn(rank, rank)
                U = U @ Q_U
                V = V @ Q_V
                W = W @ Q_W
    
    return None

def _gradient_descent(U, V, W, n, m, p, lr=0.01, max_steps=100, tol=1e-8):
    """Run gradient descent to refine the decomposition."""
    rank = U.shape[1]
    for step in range(max_steps):
        # Compute gradients efficiently
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # grad_U[i,j,r] = sum_k (U_r[i,j,r]*V_r[j,k,r]*W_r[k,i,r] - 1) * V_r[j,k,r] * W_r[k,i,r]
        
        grad_U = np.zeros_like(U_r)
        grad_V = np.zeros_like(V_r)
        grad_W = np.zeros_like(W_r)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should be 1
                    T_ijk = np.sum(U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:])
                    diff = T_ijk - 1.0
                    # d/dU[i,j,r] = diff * V_r[j,k,r] * W_r[k,i,r]
                    grad_U[i,j,:] += diff * V_r[j,k,:] * W_r[k,i,:]
                    grad_V[j,k,:] += diff * U_r[i,j,:] * W_r[k,i,:]
                    grad_W[k,i,:] += diff * U_r[i,j,:] * V_r[j,k,:]
        
        U -= lr * grad_U.reshape(n * m, rank)
        V -= lr * grad_V.reshape(m * p, rank)
        W -= lr * grad_W.reshape(p * n, rank)
        
        if step % 10 == 0:
            if _validate_fast(U, V, W, n, m, p, tol=1e-6):
                return U, V, W, True
    
    return U, V, W, _validate_fast(U, V, W, n, m, p, tol=1e-10)

def _polish(U, V, W, n, m, p, max_iter=50):
    """High-precision polish pass."""
    for _ in range(max_iter):
        if _validate_fast(U, V, W, n, m, p, tol=1e-12):
            break
        U, V, W, converged = _gradient_descent(U, V, W, n, m, p, lr=0.001, max_steps=20, tol=1e-12)
        if converged:
            break
    return U, V, W

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as default
    U, V, W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    best_result = (U, V, W)
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound up to schoolbook_rank - 1
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 2:
            break
        
        # Allocate time per rank
        time_per_rank = min(10.0, time_remaining / (schoolbook_rank - rank))
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 3:
                break
            
            # Try structured initialization
            result = _try_structured_decomposition(n, m, p, rank, seed=seed, time_budget=time_per_rank/20)
            
            # Also try direct random init + gradient descent
            rng = np.random.RandomState(seed + 1000)
            U_try = rng.randn(n * m, rank) * 0.3
            V_try = rng.randn(m * p, rank) * 0.3
            W_try = rng.randn(p * n, rank) * 0.3
            
            # Run gradient descent
            U_try, V_try, W_try, converged = _gradient_descent(
                U_try, V_try, W_try, n, m, p, 
                lr=0.02, max_steps=50, tol=1e-6
            )
            
            if converged:
                # Polish
                U_try, V_try, W_try = _polish(U_try, V_try, W_try, n, m, p)
                if _validate_fast(U_try, V_try, W_try, n, m, p, tol=1e-10):
                    # Snap to half-integers
                    U_snap = _snap_to_half_integers(U_try)
                    V_snap = _snap_to_half_integers(V_try)
                    W_snap = _snap_to_half_integers(W_try)
                    if _validate_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-10):
                        U, V, W = U_snap, V_snap, W_snap
                        best_rank = rank
                        best_result = (U, V, W)
                        found = True
                        break
                    else:
                        # Use un-snapped if it validates
                        U, V, W = U_try, V_try, W_try
                        best_rank = rank
                        best_result = (U, V, W)
                        found = True
                        break
        
        if found:
            # Found a valid decomposition at this rank, try lower ranks if time allows
            continue
    
    # Final validation
    if not _validate_fast(U, V, W, n, m, p, tol=1e-10):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
