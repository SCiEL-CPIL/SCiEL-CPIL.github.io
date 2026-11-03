"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Vectorized check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _fast_als_step(U, V, W, n, m, p):
    """
    Perform one ALS iteration with vectorized operations.
    Returns updated (U, V, W).
    """
    r = U.shape[1]
    
    # Update U: for each (i,j), solve A_mat @ u = ones(p)
    # A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
    U_new = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            # Vectorized: A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Reshape V[j*p:(j+1)*p, :] and W[::n, :] for this i
            V_slice = V[j*p:(j+1)*p, :]  # (p, r)
            W_slice = W[i::n, :]  # (p, r) - W[k*n+i] for k=0..p-1
            A_mat = V_slice * W_slice  # (p, r)
            b_vec = np.ones(p)
            try:
                u_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                U_new[i*m+j, :] = u_sol
            except:
                U_new[i*m+j, :] = U[i*m+j, :]
    U = U_new
    
    # Update V: for each (j,k), solve A_mat @ v = ones(n)
    # A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
    V_new = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            # U[i*m+j, :] for i=0..n-1
            U_slice = U[::m, j]  # (n,) - U[i*m+j] for i=0..n-1
            # W[k*n+i, :] for i=0..n-1
            W_slice = W[k*n:(k+1)*n, :]  # (n, r)
            A_mat = U_slice[:, None] * W_slice  # (n, r)
            b_vec = np.ones(n)
            try:
                v_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                V_new[j*p+k, :] = v_sol
            except:
                V_new[j*p+k, :] = V[j*p+k, :]
    V = V_new
    
    # Update W: for each (k,i), solve A_mat @ w = ones(m)
    # A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
    W_new = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            # U[i*m+j, :] for j=0..m-1
            U_slice = U[i*m:(i+1)*m, :]  # (m, r)
            # V[j*p+k, :] for j=0..m-1
            V_slice = V[::p, k]  # (m,) - V[j*p+k] for j=0..m-1
            A_mat = U_slice * V_slice[:, None]  # (m, r)
            b_vec = np.ones(m)
            try:
                w_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                W_new[k*n+i, :] = w_sol
            except:
                W_new[k*n+i, :] = W[k*n+i, :]
    W = W_new
    
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute the decomposition error."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0) ** 2
    return np.sqrt(err / (n*m*p))

def _try_rank(r, n, m, p, max_iterations=3000, seed=42, time_limit=100):
    """
    Try to find a valid decomposition of rank r using alternating least squares.
    Returns (U, V, W) as numpy arrays if found, else None.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Pure random
    U1 = rng.randn(n*m, r)
    V1 = rng.randn(m*p, r)
    W1 = rng.randn(p*n, r)
    inits.append((U1, V1, W1))
    
    # 2. Sparse random (many zeros)
    U2 = rng.binomial(1, 0.3, (n*m, r)).astype(float) * rng.randn(n*m, r)
    V2 = rng.binomial(1, 0.3, (m*p, r)).astype(float) * rng.randn(m*p, r)
    W2 = rng.binomial(1, 0.3, (p*n, r)).astype(float) * rng.randn(p*n, r)
    inits.append((U2, V2, W2))
    
    # 3. Kronecker-like structure
    if r >= n * p:
        U3 = np.zeros((n*m, r))
        V3 = np.zeros((m*p, r))
        W3 = np.zeros((p*n, r))
        for i in range(n):
            for k in range(p):
                idx = i * p + k
                if idx < r:
                    for j in range(m):
                        U3[i*m+j, idx] = 1.0
                    V3[k, idx] = 1.0
                    W3[k*n+i, idx] = 1.0
        inits.append((U3, V3, W3))
    
    # 4. Diagonal-like
    U4 = np.zeros((n*m, r))
    V4 = np.zeros((m*p, r))
    W4 = np.zeros((p*n, r))
    for idx in range(min(r, n*m*p)):
        i = idx % n
        j = (idx // n) % m
        k = (idx // (n*m)) % p
        U4[i*m+j, idx % r] = 1.0
        V4[j*p+k, idx % r] = 1.0
        W4[k*n+i, idx % r] = 1.0
    inits.append((U4, V4, W4))
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for U_init, V_init, W_init in inits:
        if time.time() - start_time > time_limit:
            break
        
        U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
        
        for it in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W = _fast_als_step(U, V, W, n, m, p)
            
            # Check convergence
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                return U, V, W
    
    # Check if best solution is valid
    if best_U is not None and _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-6):
        return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = 140  # safety margin before 150s
    
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
    
    # Known lower bound: for matrix multiplication, rank >= max(n*m, m*p, p*n)
    # For 4x5x5: max(20, 25, 20) = 25
    # But we can do better with structured decompositions
    
    # Try ranks in increasing order from a reasonable lower bound
    min_rank = max(n*m, m*p, p*n)  # 25 for 4x5x5
    # Actually, for matrix multiplication, the rank is at least the "slice rank"
    # Let's try from a lower bound
    
    # For 4x5x5, known results suggest rank around 25-30
    # Start from a lower bound and go up
    
    for target_rank in range(20, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        found = False
        num_seeds = 15 if target_rank <= 30 else 8
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            
            remaining_time = deadline - time.time()
            if remaining_time < 2:
                break
            
            time_per_try = max(1, min(30, int(remaining_time / max(1, num_seeds - seed))))
            
            result = _try_rank(target_rank, n, m, p, max_iterations=2000, seed=seed, time_limit=time_per_try)
            if result is not None:
                U_r, V_r, W_r = result
                # Verify
                if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
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
