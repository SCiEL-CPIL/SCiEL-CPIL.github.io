"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Vectorized check: compute all (i,j,k) values at once
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
    # Reshape for broadcasting
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U[i,j,:] shape (n,m,rank), V[j,k,:] shape (m,p,rank), W[k,i,:] shape (p,n,rank)
    # Need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    vals = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    if np.any(np.abs(vals - 1.0) > tol):
        return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the average absolute error of the decomposition."""
    U_reshaped = U.reshape(n, m, U.shape[1])
    V_reshaped = V.reshape(m, p, V.shape[1])
    W_reshaped = W.reshape(p, n, W.shape[1])
    vals = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.mean(np.abs(vals - 1.0))

def _try_rank(r, n, m, p, max_iterations=5000, seed=42, time_limit=100):
    """
    Try to find a valid decomposition of rank r using alternating least squares.
    Returns (U, V, W) as numpy arrays if found, else None.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Random initialization
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Fix V, W: solve for U. For each (i,j), the equation is:
        # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, r))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b_vec = np.ones(p)
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_new[i*m+j, :] = u_sol
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        U = U_new
        
        # Fix U, W: solve for V. For each (j,k), the equation is:
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, r))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b_vec = np.ones(n)
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_new[j*p+k, :] = v_sol
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        V = V_new
        
        # Fix U, V: solve for W. For each (k,i), the equation is:
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b_vec = np.ones(m)
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_new[k*n+i, :] = w_sol
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        W = W_new
        
        # Check convergence using vectorized error computation
        err = _compute_error(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            return U, V, W
    
    # Check if best solution is valid
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-6):
        return best_U, best_V, best_W
    
    return None

def _polish(U, V, W, n, m, p, max_iterations=100, tol=1e-10, time_limit=20):
    """Final high-precision polish pass with tighter tolerance."""
    start_time = time.time()
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    r = U.shape[1]
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # One full ALS cycle
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, r))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b_vec = np.ones(p)
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_new[i*m+j, :] = u_sol
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        U = U_new
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, r))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b_vec = np.ones(n)
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_new[j*p+k, :] = v_sol
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        V = V_new
        
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b_vec = np.ones(m)
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_new[k*n+i, :] = w_sol
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        W = W_new
        
        err = _compute_error(U, V, W, n, m, p)
        if err < tol:
            break
    
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = 115  # safety margin before 120s (kill at 150s)
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Known lower bound: for matrix multiplication n x m by m x p,
    # rank >= max(n*m, m*p, p*n) / something... 
    # Actually for 4x5*5x5, known results suggest rank around 20-30 might be achievable
    # Let's start from a reasonable lower bound and go up
    # Lower bound: at least max(n, m, p) * min(n, m, p) is a heuristic
    # For 4x5x5: start from rank 16 and go up
    min_rank_to_try = 16
    
    # Try ranks in increasing order
    for target_rank in range(min_rank_to_try, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts with different seeds
        found = False
        for seed in range(15):
            if time.time() > deadline:
                break
            remaining_time = deadline - time.time()
            if remaining_time < 5:
                break
            
            # Allocate time budget per attempt
            time_per_attempt = min(remaining_time / max(1, 15 - seed), 30)
            
            result = _try_rank(target_rank, n, m, p, max_iterations=3000, seed=seed, time_limit=time_per_attempt)
            if result is not None:
                U_r, V_r, W_r = result
                # Verify with tight tolerance
                if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    # Try to polish for higher precision
                    polished = _polish(U_r, V_r, W_r, n, m, p, max_iterations=50, tol=1e-10, time_limit=10)
                    if polished is not None:
                        U_r, V_r, W_r = polished
                    
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
