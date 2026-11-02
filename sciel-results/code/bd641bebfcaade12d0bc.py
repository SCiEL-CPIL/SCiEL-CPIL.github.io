import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq
from itertools import product
import random

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 135.0  # safety margin before 150s kill

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    All other entries are 0.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _tensor_error(U, V, W, T):
    """
    Compute ||U V^T W^T - T||_F^2 where U,V,W are 2D arrays (rows x rank).
    The decomposition is T = sum_r U[:,r] V[:,r] W[:,r]^T
    which in matrix form for the first two indices is:
    T[:,:,c] = U @ (V * W[:,c])^T for each c? No.
    
    Actually T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    So for fixed k, the matrix M_k of size (n*m x m*p) is U @ diag(V[:,k]??) no.
    
    Let's reshape: T is (A, B, C).
    The decomposition means T = sum_r U[:,r] outer V[:,r] outer W[:,r].
    We can compute the error by iterating over r, or by using einsum.
    np.einsum('ar,br,cr->abc', U, V, W)
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((reconstructed - T) ** 2)

def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _structured_init(n, m, p, rank, seed):
    """
    Create structured initializations that might be closer to a low-rank solution.
    """
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Random
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    # Normalize
    for mat in [U, V, W]:
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms
    
    return U, V, W

def _als_step(U, V, W, T, n, m, p):
    """
    Perform one ALS iteration.
    """
    # Update U
    V_rep = V[:, np.newaxis, :]  # (m*p, 1, rank)
    W_rep = W[np.newaxis, :, :]  # (1, p*n, rank)
    MW = V_rep * W_rep  # (m*p, p*n, rank)
    MW_flat = MW.reshape(m*p*p*n, rank)
    
    T_flat = T.reshape(n*m, m*p*p*n)
    U_new, _, _, _ = lstsq(MW_flat, T_flat.T, cond=None)
    U = U_new.T  # (n*m, rank)
    
    # Update V
    U_rep = U[:, np.newaxis, :]  # (n*m, 1, rank)
    W_rep = W[np.newaxis, :, :]  # (1, p*n, rank)
    MW = U_rep * W_rep  # (n*m, p*n, rank)
    MW_flat = MW.reshape(n*m*p*n, rank)
    
    T_flat_V = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
    V_new, _, _, _ = lstsq(MW_flat, T_flat_V.T, cond=None)
    V = V_new.T  # (m*p, rank)
    
    # Update W
    U_rep = U[:, np.newaxis, :]  # (n*m, 1, rank)
    V_rep = V[np.newaxis, :, :]  # (1, m*p, rank)
    MW = U_rep * V_rep  # (n*m, m*p, rank)
    MW_flat = MW.reshape(n*m*m*p, rank)
    
    T_flat_W = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
    W_new, _, _, _ = lstsq(MW_flat, T_flat_W.T, cond=None)
    W = W_new.T  # (p*n, rank)
    
    return U, V, W

def _try_rank(n, m, p, rank, T, seed=0, time_limit=10.0):
    """
    Attempt to find a decomposition of the given rank.
    Returns (U, V, W, error) if successful, or None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U, V, W = _structured_init(n, m, p, rank, seed)
    
    best_err = _tensor_error(U, V, W, T)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Alternating Least Squares (ALS)
    start_time = time.time()
    iterations = 0
    max_iterations = 3000
    
    while iterations < max_iterations:
        if time.time() - start_time > time_limit:
            break
            
        U, V, W = _als_step(U, V, W, T, n, m, p)
        iterations += 1
        
        err = _tensor_error(U, V, W, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        # Convergence check
        if best_err < 1e-12:
            break
    
    return best_U, best_V, best_W, best_err

def _gradient_descent_polish(U, V, W, T, n, m, p, rank, time_limit=5.0):
    """
    Use gradient descent to polish a near-converged solution.
    """
    start_time = time.time()
    lr = 0.01
    best_err = _tensor_error(U, V, W, T)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradients
        # dT/dr = sum of contributions
        # For simplicity, use finite differences or analytical gradients
        
        # Analytical gradient for U:
        # d/dU[i,r] = sum_{b,c} T[i,b,c] * V[b,r] * W[c,r] - sum_{b,c} (sum_r' U[i,r']V[b,r']W[c,r']) * V[b,r] * W[c,r]
        
        # This is complex, so let's use a simple approach: if we're close, just do more ALS
        
        U, V, W = _als_step(U, V, W, T, n, m, p)
        err = _tensor_error(U, V, W, T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-14:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_solution(n, m, p)
    best_err = 0.0
    T = _build_tensor(n, m, p)
    
    # Try to reduce rank from n*m*p - 1 downwards
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25.
    # But known results might be lower.
    # Let's try ranks: 49, 48, 47, ..., 25.
    
    # To save time, we'll try a few promising ones with multiple seeds.
    # Ranks to try: 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25
    
    # We'll be more aggressive: try all ranks from 49 down to 25, but with limited time per rank.
    
    # First, let's see if we can find any rank < 50.
    # If not, return schoolbook.
    
    found_rank = n * m * p
    found_U, found_V, found_W = U, V, W
    
    # Try ranks in decreasing order
    for rank in range(n*m*p - 1, 24, -1):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
            
        # Try multiple seeds for this rank
        success = False
        for seed in range(3):
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                break
                
            time_limit = 8.0
            try:
                U_new, V_new, W_new, err = _try_rank(n, m, p, rank, T, seed=seed, time_limit=time_limit)
                if err < 1e-10:
                    # Check for exactness with tighter tolerance
                    if _tensor_error(U_new, V_new, W_new, T) < 1e-12:
                        # Polish with gradient descent
                        U_pol, V_pol, W_pol, err_pol = _gradient_descent_polish(
                            U_new, V_new, W_new, T, n, m, p, rank, time_limit=3.0
                        )
                        if err_pol < 1e-12:
                            U, V, W = U_pol, V_pol, W_pol
                            found_rank = rank
                            found_U, found_V, found_W = U, V, W
                            success = True
                            break
            except Exception:
                continue
                
        if success:
            # Found a solution for this rank. Continue to try lower ranks.
            continue
        else:
            # Failed to find solution for this rank.
            # Try the next lower rank only if we have time.
            # If we've already found a rank, keep trying lower.
            # If we haven't found any rank < 50, break after a few failures.
            if found_rank == n * m * p:
                # Haven't found any improvement, break to save time
                break
            else:
                # Already found an improvement, keep trying for better
                continue
    
    # Return the best found
    U_list = found_U.tolist()
    V_list = found_V.tolist()
    W_list = found_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
