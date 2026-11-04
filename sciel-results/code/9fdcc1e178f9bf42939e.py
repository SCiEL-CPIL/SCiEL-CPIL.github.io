"""
Matrix multiplication 3x4x6 - rank-minimal exact tensor decomposition.
Uses alternating projection with integer/half-integer constraints to search
for decompositions of rank below the schoolbook rank of 72.
"""
import numpy as np
import random
import sys
from itertools import product

DIMENSIONS = [3, 4, 6]


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    # Build the full tensor from the factors
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Faster verification using numpy broadcasting.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can do this by building the tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    # This is a contraction over r
    # We can compute: T[i,j,k] = sum_r U_reshaped[i,j,r] * (V_reshaped[j,k,r] * W_reshaped[k,i,r])
    
    # Let's use einsum
    # U_reshaped: (n, m, rank)
    # V_reshaped: (m, p, rank)
    # W_reshaped: (p, n, rank)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Use np.einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if all entries are 1
    if np.allclose(T, 1.0, atol=tol):
        return True
    return False


def _try_rank(rank, n, m, p, max_iters=200, seed=0, tol=1e-9):
    """
    Try to find a decomposition of the given rank using alternating projections.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Scale to make the decomposition work approximately
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Normalize each factor
    for _ in range(1000):
        # Compute current tensor
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Error
        err = np.sum((T - 1.0) ** 2)
        
        if err < tol:
            # Try to snap to integer/half-integer values
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify_decomposition_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                return U_snap, V_snap, W_snap
        
        # Alternating projection
        # Update U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        # This is a linear system in U[i*m+j, :]
        
        # For fixed (i,j), we need: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
        # Let A_{k,r} = V[j*p+k, r] * W[k*n+i, r]
        # Then A @ U[i*m+j, :] = 1 (vector of ones of length p)
        
        for i in range(n):
            for j in range(m):
                # Build A: (p, rank)
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                
                # Solve A @ x = 1
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                except:
                    x = np.zeros(rank)
                
                U[i * m + j, :] = x
        
        # Similarly update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                except:
                    x = np.zeros(rank)
                
                V[j * p + k, :] = x
        
        # Similarly update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except:
                    x = np.zeros(rank)
                
                W[k * n + i, :] = x
    
    # Final check
    if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if _verify_decomposition_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
            return U_snap, V_snap, W_snap
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Schoolbook decomposition as fallback
    U = np.zeros((n * m, schoolbook_rank), dtype=float)
    V = np.zeros((m * p, schoolbook_rank), dtype=float)
    W = np.zeros((p * n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    assert _verify_decomposition_fast(U, V, W, n, m, p), "Schoolbook decomposition failed"
    
    # Try lower ranks using alternating projection
    # Based on known results for similar sizes, try ranks from 45 to 71
    for trial_rank in range(45, min(72, schoolbook_rank)):
        for seed in range(10):
            result = _try_rank(trial_rank, n, m, p, max_iters=100, seed=seed)
            if result is not None:
                U_found, V_found, W_found = result
                if _verify_decomposition_fast(U_found, V_found, W_found, n, m, p):
                    return {"U": U_found.tolist(), "V": V_found.tolist(), "W": W_found.tolist()}
    
    # Return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
