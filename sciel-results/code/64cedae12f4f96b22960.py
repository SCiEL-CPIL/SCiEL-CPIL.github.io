"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _compute_tensor_residual(U, V, W, n, m, p):
    """Compute the tensor residual R[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1"""
    R = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # R[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        for i in range(n):
            for j in range(m):
                R[i,j,:] += uc[i,j] * vc[j,:] * wc[:,i]
    R -= 1.0
    return R

def _alternating_minimization(n, m, p, rank, max_iters=500, tol=1e-10, seed=42, max_time=30.0):
    """Alternating least squares for tensor decomposition."""
    rng = np.random.default_rng(seed)
    
    # Initialize
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > max_time:
            break
        
        # Update U: for each (i,j), minimize sum_k (R[i,j,k] + sum_{c'} U[i,j,c']*V[j,k,c']*W[k,i,c'])^2
        # This is equivalent to solving a linear system
        # For fixed V, W: minimize sum_{j,k} (1 - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
        # w.r.t. U[i,j,:]
        
        # Build the linear system for U
        # Let A_U[i,j] be a matrix where A_U[i,j][:,c] = sum_k V[j,k,c]*W[k,i,c]
        # Then we want: U[i,j,:] @ A_U[i,j] = 1 (scalar)
        # But it's underdetermined if rank > 1, so we use least squares
        
        # More efficient: for each i,j, the target is 1, and the "design matrix" is
        # D[i,j,c] = sum_k V[j*p+k, c] * W[k*n+i, c]
        # Then U[i*m+j, :] should satisfy U[i*m+j, :] @ D[i,j,:] = 1
        
        # Actually, let me think again. The residual for entry (i,j,k) is:
        # r[i,j,k] = 1 - sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # For fixed V, W, minimizing sum_{j,k} r[i,j,k]^2 w.r.t. U[i*m+j, :]
        # gives: U[i*m+j, :] = (sum_{j,k} ... ) 
        # This is a linear least squares: min ||M @ u - 1||^2 where M is (m*p) x rank
        
        # Let's vectorize over i
        for i in range(n):
            # For this i, we need to update U[i*m:(i+1)*m, :]
            # The constraint involves all j, k
            # Build matrix: for each j, the contribution to the residual sum is
            # sum_k (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            
            # Let's define for each j: b_j = sum_k V[j*p+k,c]*W[k*n+i,c] for each c
            # Then residual for (i,j,k) = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            
            # The full residual sum over j,k:
            # sum_j sum_k (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            
            # This is separable in j! For each j independently:
            # min sum_k (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            
            # Let A_j be a (p x rank) matrix: A_j[k,c] = V[j*p+k,c]*W[k*n+i,c]
            # Let u_j = U[i*m+j, :] (rank-vector)
            # min ||A_j @ u_j - 1||^2 (where 1 is a p-vector of ones)
            
            # Solution: u_j = (A_j^T A_j)^{-1} A_j^T 1
            
            for j in range(m):
                # A_j[k,c] = V[j*p+k, c] * W[k*n+i, c]
                Vj = V[j*p:(j+1)*p, :]  # (p, rank)
                Wi = W[:, i]  # (p,) - W[k*n+i] for all k
                # A_j = Vj * Wi[None, :]  # (p, rank)
                A_j = Vj * Wi[None, :]  # (p, rank)
                b_j = np.ones(p)
                
                # Solve A_j @ u_j = b_j in least squares
                try:
                    u_j, _, _, _ = np.linalg.lstsq(A_j, b_j, rcond=None)
                except:
                    u_j = np.ones(rank) / rank
                
                U[i*m+j, :] = u_j
        
        # Update V similarly
        for j in range(m):
            # For fixed U, W: min sum_{i,k} (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            # For each k: min sum_i (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            # Let A_k be (n x rank): A_k[i,c] = U[i*m+j,c]*W[k*n+i,c]
            # min ||A_k @ v_k - 1||^2 where v_k = V[j*p+k, :]
            
            for k in range(p):
                Uk = U[:, j]  # (n,) - U[i*m+j] for all i
                Wk = W[k*n:(k+1)*n, :]  # (n, rank)
                A_k = Uk[:, None] * Wk  # (n, rank)
                b_k = np.ones(n)
                
                try:
                    v_k, _, _, _ = np.linalg.lstsq(A_k, b_k, rcond=None)
                except:
                    v_k = np.ones(rank) / rank
                
                V[j*p+k, :] = v_k
        
        # Update W similarly
        for k in range(p):
            # For fixed U, V: min sum_{i,j} (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            # For each i: min sum_j (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            # Let A_i be (m x rank): A_i[j,c] = U[i*m+j,c]*V[j*p+k,c]
            # min ||A_i @ w_i - 1||^2 where w_i = W[k*n+i, :]
            
            for i in range(n):
                Ui = U[i*m:(i+1)*m, :]  # (m, rank)
                Vk = V[:, k]  # (m,) - V[j*p+k] for all j
                A_i = Ui * Vk[None, :]  # (m, rank)
                b_i = np.ones(m)
                
                try:
                    w_i, _, _, _ = np.linalg.lstsq(A_i, b_i, rcond=None)
                except:
                    w_i = np.ones(rank) / rank
                
                W[k*n+i, :] = w_i
        
        # Check convergence
        R = _compute_tensor_residual(U, V, W, n, m, p)
        if np.max(np.abs(R)) < tol:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Start with schoolbook decomposition
    U0, V0, W0 = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = n * m * p
    
    # Try alternating minimization for various ranks
    # Lower bound: at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    min_rank = max(n * m, m * p, p * n)
    
    # Try ranks from min_rank to best_rank - 1
    ranks_to_try = list(range(min_rank, best_rank))
    random.shuffle(ranks_to_try)
    
    search_deadline = time.time() + 90
    
    for rank in ranks_to_try:
        if time.time() > search_deadline:
            break
        if rank >= best_rank:
            continue
        
        # Try multiple seeds for robustness
        found = False
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > search_deadline:
                break
            U, V, W = _alternating_minimization(n, m, p, rank, max_iters=200, tol=1e-10, seed=seed, max_time=15.0)
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
        
        if found and rank <= min_rank + 2:
            break
    
    return best_result
# EVOLVE-BLOCK-END
