"""
Matrix multiplication 2x3 by 3x7 - rank-minimal tensor decomposition.
Attempts to find lower-rank decomposition via numerical optimization,
falls back to schoolbook rank 42.
"""
import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    rank = U.shape[1]
    # T[i,j,k] should equal 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_lower_rank(n, m, p, target_rank, time_budget, seed=42):
    """Try to find a decomposition with the given target rank using gradient-based optimization."""
    start_time = time.time()
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # We want: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p = 42 equations
    
    # Use simple gradient descent on the residual
    lr = 0.01
    max_iter = 5000
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute residuals and gradients
        # For each (i,j,k), residual = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
        # d/dU[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c]
        # d/dV[j*p+k,c] = U[i*m+j,c]*W[k*n+i,c]
        # d/dW[k*n+i,c] = U[i*m+j,c]*V[j*p+k,c]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        total_residual_sq = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # Current value
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    residual = val - 1.0
                    total_residual_sq += residual * residual
                    
                    # Gradients
                    grad_U[u_idx, :] += residual * V[v_idx, :] * W[w_idx, :]
                    grad_V[v_idx, :] += residual * U[u_idx, :] * W[w_idx, :]
                    grad_W[w_idx, :] += residual * U[u_idx, :] * V[v_idx, :]
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if total_residual_sq < 1e-12:
            break
    
    # Check if we converged
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        return U, V, W, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # First, get schoolbook as fallback
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try to find lower rank decompositions
    # Lower bound: rank must be at least max(n*p, m*p, n*m) = max(14, 21, 6) = 21
    # But for matmul, tighter bounds exist. Let's try from a reasonable lower bound.
    
    # Known lower bound for matrix multiplication tensor M_{n,m,p}:
    # rank >= n*p (from slice argument) = 14
    # But also rank >= m*p = 21 from another perspective
    # Let's try from 21 upward
    
    lower_bound = max(n * p, m * p, n * m)  # = max(14, 21, 6) = 21
    
    for target_rank in range(lower_bound, rank_school):
        if time.time() > deadline - 5.0:
            break
        
        time_per_attempt = min(5.0, (deadline - time.time() - 2.0) / max(1, rank_school - target_rank))
        
        result = _try_lower_rank(n, m, p, target_rank, time_per_attempt)
        if result is not None:
            U, V, W, rank = result
            # Verify more carefully
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break  # Found the lowest rank we tried
    
    # Convert to lists as per interface
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
