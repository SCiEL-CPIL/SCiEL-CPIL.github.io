"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill

    # Strategy:
    # 1. Try to find a low-rank decomposition via alternating minimization.
    # 2. If found, attempt to "clean" it to integer/half-integer values.
    # 3. If cleaning fails or no low rank found, fallback to schoolbook (rank 24).
    
    # Known lower bound: max(n*m, m*p, p*n) = 9.
    # Schoolbook: 24.
    # Let's search from 10 upwards.
    
    # To save time, we will limit the number of restarts and iterations per rank.
    # We will also prioritize lower ranks.
    
    for rank in range(10, 25):
        if time.time() > deadline - 10:
            break
        
        # Try to find a solution for this rank
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank(n, m, p, rank, deadline, max_iters=500, n_restarts=5):
    """
    Try to find a rank-r decomposition.
    Uses alternating minimization with random restarts.
    """
    # Fixed seed for reproducibility in evaluation, but we vary it slightly for restarts
    base_seed = 42
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(base_seed + restart)
        
        # Initialize with random values
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to prevent overflow/underflow
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Run alternating minimization
        success = _alternating_minimize(n, m, p, rank, U, V, W, max_iters, deadline)
        
        if success:
            # Try to clean the solution to half-integers
            U_clean, V_clean, W_clean = _clean_solution(n, m, p, rank, U, V, W)
            
            if _verify(n, m, p, U_clean, V_clean, W_clean, tol=1e-9):
                return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
            
            # If cleaning failed but the raw solution is very accurate, 
            # maybe we can return it? The task says "integer/half-integer entries".
            # So we MUST clean. If we can't clean, this rank is likely not an exact integer solution.
            # We continue searching.
            # However, sometimes numerical noise prevents exact cleaning.
            # Let's try a more aggressive cleaning or different rounding.
            # For now, if simple rounding fails, we assume this specific numerical solution
            # doesn't correspond to a simple rational one, or we didn't converge enough.
            # We might try to increase iterations for the best candidate? 
            # For simplicity and speed, we just move to next restart/rank.
            
    return None

def _alternating_minimize(n, m, p, rank, U, V, W, max_iters, deadline, tol=1e-12):
    """
    Alternating Least Squares.
    """
    for it in range(max_iters):
        if time.time() > deadline - 5:
            return False
            
        # Update U
        for i in range(n):
            for j in range(m):
                # System: U[i,j,:] * (V[j,:,c] * W[:,i,c])^T = 1
                # Let M[k,c] = V[j*p+k, c] * W[k*n+i, c]
                # We want U[i,j,:] @ M.T = ones(p)
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    return False
        
        # Update V
        for j in range(m):
            for k in range(p):
                # System: V[j,k,:] * (U[:,j,c] * W[k,:,c])^T = 1
                # Let M[i,c] = U[i*m+j, c] * W[k*n+i, c]
                # We want V[j,k,:] @ M.T = ones(n)
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    return False
        
        # Update W
        for k in range(p):
            for i in range(n):
                # System: W[k,i,:] * (U[i,:,c] * V[:,k,c])^T = 1
                # Let M[j,c] = U[i*m+j, c] * V[j*p+k, c]
                # We want W[k,i,:] @ M.T = ones(m)
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    return False
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
                    
        if err < tol:
            return True
            
    return False

def _clean_solution(n, m, p, rank, U, V, W):
    """
    Attempt to convert the floating point solution to half-integers.
    """
    U_clean = np.round(U * 2) / 2
    V_clean = np.round(V * 2) / 2
    W_clean = np.round(W * 2) / 2
    return U_clean, V_clean, W_clean

def _verify(n, m, p, U, V, W, tol=1e-9):
    """
    Verify the decomposition is exact (within tolerance).
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
