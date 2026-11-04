"""Matrix multiplication 2x5x5 rank minimization via structured algebraic search."""
DIMENSIONS = [2, 5, 5]

import numpy as np
import time
import random

def _build_tensor():
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    where T[i*m+j, j*p+k, k*n+i] = 1.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _structured_init(rank, n, m, p, seed=None):
    """
    Create structured initialization exploiting the block structure of matrix multiplication.
    For 2x5x5, we can group terms by (i,j) pairs and try to share across k.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with random values but with some structure
    U = rng.randn(nm, rank) * 0.1
    V = rng.randn(mp, rank) * 0.1
    W = rng.randn(pn, rank) * 0.1
    
    # Add structured components: for each (i,j) pair, try to have terms that cover multiple k
    # This exploits the fact that T[i*m+j, j*p+k, k*n+i] = 1 for all k
    for i in range(n):
        for j in range(m):
            # Assign a few rank terms to handle this (i,j) slice
            start = (i * m + j) * (rank // (n * m))
            end = start + max(1, rank // (n * m))
            for r in range(start, min(end, rank)):
                # These terms should contribute to all k for this (i,j)
                U[i * m + j, r] += 1.0
                for k in range(p):
                    V[j * p + k, r] += 1.0
                    W[k * n + i, r] += 1.0
    
    return U, V, W

def _als_decompose(T, rank, max_iter=200, tol=1e-9, U_init=None, V_init=None, W_init=None, deadline=None):
    """
    Try to find a decomposition of rank `rank` using Alternating Least Squares (ALS).
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    if U_init is not None and V_init is not None and W_init is not None:
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
    else:
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
    
    prev_err = np.inf
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for r in range(rank):
            num = np.einsum('abc,b,c->a', T, V[:, r], W[:, r])
            den = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if den > 1e-12:
                U[:, r] = num / den
        
        # Update V
        for r in range(rank):
            num = np.einsum('abc,a,c->b', T, U[:, r], W[:, r])
            den = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if den > 1e-12:
                V[:, r] = num / den
        
        # Update W
        for r in range(rank):
            num = np.einsum('abc,a,b->c', T, U[:, r], V[:, r])
            den = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
            if den > 1e-12:
                W[:, r] = num / den
        
        # Compute error
        recon = np.zeros_like(T)
        for r in range(rank):
            recon += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        err = np.linalg.norm(recon - T)
        
        # Check for convergence or stagnation
        if err < tol:
            return U, V, W, True
        if abs(prev_err - err) < 1e-12:
            break
        prev_err = err
    
    return U, V, W, False

def _polish_decomposition(U, V, W, T, max_iter=200, tol=1e-12, deadline=None):
    """
    High-precision polish pass using ALS with tighter tolerance.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for r in range(rank):
            num = np.einsum('abc,b,c->a', T, V[:, r], W[:, r])
            den = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if den > 1e-14:
                U[:, r] = num / den
        
        # Update V
        for r in range(rank):
            num = np.einsum('abc,a,c->b', T, U[:, r], W[:, r])
            den = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if den > 1e-14:
                V[:, r] = num / den
        
        # Update W
        for r in range(rank):
            num = np.einsum('abc,a,b->c', T, U[:, r], V[:, r])
            den = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
            if den > 1e-14:
                W[:, r] = num / den
        
        # Compute error
        recon = np.zeros_like(T)
        for r in range(rank):
            recon += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        err = np.linalg.norm(recon - T)
        if err < tol:
            break
    
    return U, V, W

def _snap_and_verify(U, V, W, T, tol=1e-9):
    """
    Snap the matrices to half-integers and verify the decomposition.
    """
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    
    recon = np.zeros_like(T)
    for r in range(U.shape[1]):
        recon += U_s[:, r][:, None, None] * V_s[:, r][None, :, None] * W_s[:, r][None, None, :]
    
    if np.allclose(recon, T, atol=tol):
        return U_s, V_s, W_s, True
    return None, None, None, False

def _try_rank_with_structured_search(T, rank, n, m, p, deadline, max_restarts=30):
    """
    Try to find a decomposition of the given rank using structured initialization and multiple strategies.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Strategy 1: Structured initialization
        if restart % 3 == 0:
            U_init, V_init, W_init = _structured_init(rank, n, m, p, seed=restart * 1000)
            U, V, W, success = _als_decompose(T, rank, max_iter=150, tol=1e-9, U_init=U_init, V_init=V_init, W_init=W_init, deadline=deadline)
        # Strategy 2: Random initialization with scaling
        elif restart % 3 == 1:
            scale = 0.1 + 0.1 * (restart % 10)
            U_init = np.random.randn(nm, rank) * scale
            V_init = np.random.randn(mp, rank) * scale
            W_init = np.random.randn(pn, rank) * scale
            U, V, W, success = _als_decompose(T, rank, max_iter=150, tol=1e-9, U_init=U_init, V_init=V_init, W_init=W_init, deadline=deadline)
        # Strategy 3: Pure random
        else:
            U, V, W, success = _als_decompose(T, rank, max_iter=150, tol=1e-9, deadline=deadline)
        
        if success:
            # Polish with tight tolerance
            U, V, W = _polish_decomposition(U, V, W, T, max_iter=100, tol=1e-14, deadline=deadline)
            U_s, V_s, W_s, valid = _snap_and_verify(U, V, W, T)
            if valid:
                return U_s, V_s, W_s, True
    
    return None, None, None, False

def _schoolbook_decomposition():
    """Return the standard schoolbook decomposition of rank 50."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with shapes (10, rank), (25, rank), (10, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    T = _build_tensor()
    n, m, p = DIMENSIONS
    
    # Try ranks from 40 to 50 with increasing effort
    for rank in range(40, 51):
        if time.time() > deadline - 10:
            break
        
        # Determine how many restarts to allow based on remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # More restarts for lower ranks (harder to find)
        if rank <= 45:
            max_restarts = max(10, min(80, int(remaining / 1.5)))
        else:
            max_restarts = max(5, min(40, int(remaining / 2)))
        
        U_s, V_s, W_s, found = _try_rank_with_structured_search(T, rank, n, m, p, deadline, max_restarts=max_restarts)
        
        if found:
            return {
                "U": U_s.tolist(),
                "V": V_s.tolist(),
                "W": W_s.tolist()
            }
    
    # Fallback to schoolbook decomposition (rank 50)
    U, V, W = _schoolbook_decomposition()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
