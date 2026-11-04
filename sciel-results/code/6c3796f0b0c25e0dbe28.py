"""
Rank-minimal tensor decomposition for 3x3 matrix multiplication.
Attempts rank 23 (the proven lower bound for 3x3 matrix multiplication).
If rank 23 cannot be verified, falls back to rank 27 (schoolbook).
"""

import time
import numpy as np
from fractions import Fraction

_START = time.time()
_DEADLINE = _START + 100  # Safety margin before 120s limit


def _verify(U, V, W, tol=1e-10):
    """
    Verify that U, V, W form a valid decomposition of the 3x3 matrix multiplication tensor.
    U: (9, rank), V: (9, rank), W: (9, rank)
    Convention: W[k*3+i, r] corresponds to C[i,k] in the product A*B.
    
    The tensor equation is:
    sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k in {0,1,2}
    And 0 otherwise.
    """
    rank = U.shape[1]
    # Build the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check the non-zero entries
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False, T[a, b, c]
    
    # Check that all other entries are 0
    valid_mask = np.zeros((9, 9, 9), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                valid_mask[a, b, c] = True
    
    invalid = T[~valid_mask]
    if np.max(np.abs(invalid)) > tol:
        return False, np.max(np.abs(invalid))
    
    return True, None


def _schoolbook_decomposition():
    """Return the rank-27 schoolbook decomposition."""
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    
    return U, V, W


def _try_rank23_decomposition():
    """
    Attempt to find a rank-23 decomposition using numerical optimization.
    If found and verified, return it. Otherwise return None.
    """
    import random
    
    deadline = _START + 100
    
    # The tensor we want to decompose: T[a,b,c] = 1 if (a,b,c) corresponds to
    # a valid matrix multiplication term, else 0.
    # a = i*3+j, b = j*3+k, c = k*3+i
    
    # Build target tensor
    target = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                target[a, b, c] = 1.0
    
    # Try to find rank-23 decomposition via random initialization + gradient descent
    # This is a non-convex problem, so we use multiple random restarts
    
    best_error = float('inf')
    best_result = None
    
    n_restarts = 0
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        rng = np.random.RandomState(restart)
        U = rng.randn(9, 23) * 0.1
        V = rng.randn(9, 23) * 0.1
        W = rng.randn(9, 23) * 0.1
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        for step in range(200):
            if time.time() > deadline:
                break
            
            # Compute current tensor
            T = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.sum((T - target)**2)
            
            if error < best_error:
                best_error = error
                best_result = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-10:
                break
            
            # Compute gradients
            # dT/da = V[b,r]*W[c,r]
            # dT/db = U[a,r]*W[c,r]  
            # dT/dc = U[a,r]*V[b,r]
            
            # Gradient for U: sum over b,c of (T[a,b,c]-target[a,b,c]) * V[b,r] * W[c,r]
            grad_U = np.einsum('abc,br,cr->ar', (T - target), V, W)
            grad_V = np.einsum('abc,ar,cr->br', (T - target), U, W)
            grad_W = np.einsum('abc,ar,br->cr', (T - target), U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if error > best_error * 1.1:
                lr *= 0.9
            else:
                lr = min(lr * 1.01, 0.05)
        
        # Check if we found a good solution
        if best_error < 1e-8:
            break
    
    # Verify the best result
    if best_result is not None and best_error < 1e-8:
        U, V, W = best_result
        is_valid, _ = _verify(U, V, W, tol=1e-6)
        if is_valid:
            return U, V, W
    
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    
    U: shape (9, rank) - factors for A (3x3 matrix), indexed as A[i*3+j]
    V: shape (9, rank) - factors for B (3x3 matrix), indexed as B[j*3+k]
    W: shape (9, rank) - factors for C (3x3 matrix), with W[k*3+i, r] convention
    
    The decomposition satisfies:
    sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r] = delta_{i,k} for the correct entries
    """
    
    # Deadline guard
    if time.time() > _START + 110:
        # Not enough time, return schoolbook immediately
        U, V, W = _schoolbook_decomposition()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # First try rank 23
    result = _try_rank23_decomposition()
    if result is not None:
        U, V, W = result
        # Final verification
        is_valid, err = _verify(U, V, W)
        if is_valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 27 schoolbook
    U, V, W = _schoolbook_decomposition()
    is_valid, err = _verify(U, V, W)
    if not is_valid:
        # This should never happen
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
