"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _schoolbook_decomposition():
    """Return the schoolbook (rank 100) decomposition."""
    n, m, p = DIMENSIONS  # 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    rank = n * m * p  # 100
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-8):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    # Reconstruct the tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_integer(arr, tol=1e-6):
    """Check if all entries are half-integers (i.e., 2*x is an integer)."""
    doubled = arr * 2
    rounded = np.round(doubled)
    return np.all(np.abs(doubled - rounded) < tol)

def _try_strassen_embedded():
    """
    Try to use Strassen's algorithm for a 2x2 block decomposition.
    For 4x5 x 5x5, we can try to decompose into 2x2 blocks of matrices.
    But the dimensions don't divide evenly, so this is complex.
    
    Alternative: use known lower bounds and small-rank constructions.
    For now, return None to indicate failure.
    """
    return None

def _try_random_optimization(rank, max_time=20):
    """
    Try to find a decomposition of given rank using random initialization and gradient descent.
    This is unlikely to succeed for ranks significantly below schoolbook, but we try.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start = time.time()
    deadline = start + max_time
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + attempt * 13 + rank)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        lr = 0.01
        for it in range(2000):
            if time.time() > deadline:
                break
            
            # Compute residual and gradients
            max_err = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = val - 1.0
                        max_err = max(max_err, abs(err))
                        
                        grad_U[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :] * err
                        grad_V[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :] * err
                        grad_W[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :] * err
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if it % 200 == 0 and it > 0:
                lr *= 0.95
        
        # Check if we found a valid solution
        if max_err < 1e-6:
            # Try to round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if _is_half_integer(U_round) and _is_half_integer(V_round) and _is_half_integer(W_round):
                if _verify_decomposition(U_round, V_round, W_round, tol=1e-6):
                    return U_round, V_round, W_round
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook_decomposition()
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = 100
    
    # Try to find better solutions with lower rank
    # For 4x5x5, the theoretical lower bound is not well-known, but schoolbook is 100.
    # Known results: rank(4x4x4) <= 49 (using Strassen embedding), but for 4x5x5 it's less clear.
    
    # Try ranks from 99 down to 80 (limit by time)
    for rank in range(99, 79, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_random_optimization(rank, max_time=15)
        if result is not None:
            U, V, W = result
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = rank
            break
    
    return best_result
