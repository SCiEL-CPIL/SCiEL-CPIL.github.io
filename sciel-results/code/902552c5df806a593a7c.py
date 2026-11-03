import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _find_lower_rank(n, m, p, target_rank, deadline, seed=42):
    """
    Try to find a decomposition of rank target_rank using ALS (Alternating Least Squares).
    """
    rng = np.random.RandomState(seed)
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n_m, m_p, p_n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute index mappings
    idx_U = []  # (i, j) -> row index in U
    idx_V = []  # (j, k) -> row index in V
    idx_W = []  # (k, i) -> row index in W
    for i in range(n):
        for j in range(m):
            idx_U.append((i, j))
    for j in range(m):
        for k in range(p):
            idx_V.append((j, k))
    for k in range(p):
        for i in range(n):
            idx_W.append((k, i))
    
    # Try multiple random initializations
    n_attempts = 20
    max_iters = 100
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Random initialization with small values
        U = rng.randn(n_m, target_rank) * 0.1
        V = rng.randn(m_p, target_rank) * 0.1
        W = rng.randn(p_n, target_rank) * 0.1
        
        prev_residual = np.inf
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None
            
            # --- Update U ---
            # For each row (i,j) of U, solve:
            # T[i*m+j, :, :] = sum_r U[i*m+j, r] * (V[:, r] * W[:, r] with appropriate indexing)
            # Actually: T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
            # For fixed (i,j), we need: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # This is a linear system in U[i*m+j, :]:
            # For each (k), the coefficient vector is [V[j*p+k,r] * W[k*n+i,r] for r in range(target_rank)]
            # and the target is T[i*m+j, j*p+k, k*n+i]
            
            for i in range(n):
                for j in range(m):
                    # Build design matrix A of shape (p, target_rank)
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A_u = np.zeros((p, target_rank))
                    b_u = np.zeros(p)
                    for k in range(p):
                        A_u[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b_u[k] = T[i*m+j, j*p+k, k*n+i]
                    
                    # Solve least squares: A_u @ U[i*m+j, :] = b_u
                    try:
                        U[i*m+j, :], residuals, rank_u, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
                    except:
                        pass
            
            # --- Update V ---
            # For fixed (j,k): T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Linear system in V[j*p+k, :]:
            # For each (i), coefficient: U[i*m+j,r] * W[k*n+i,r], target: T[i*m+j, j*p+k, k*n+i]
            
            for j in range(m):
                for k in range(p):
                    A_v = np.zeros((n, target_rank))
                    b_v = np.zeros(n)
                    for i in range(n):
                        A_v[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b_v[i] = T[i*m+j, j*p+k, k*n+i]
                    
                    try:
                        V[j*p+k, :], residuals, rank_v, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
                    except:
                        pass
            
            # --- Update W ---
            # For fixed (k,i): T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Linear system in W[k*n+i, :]:
            # For each (j), coefficient: U[i*m+j,r] * V[j*p+k,r], target: T[i*m+j, j*p+k, k*n+i]
            
            for k in range(p):
                for i in range(n):
                    A_w = np.zeros((m, target_rank))
                    b_w = np.zeros(m)
                    for j in range(m):
                        A_w[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b_w[j] = T[i*m+j, j*p+k, k*n+i]
                    
                    try:
                        W[k*n+i, :], residuals, rank_w, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                    except:
                        pass
            
            # Check convergence
            T_decomp = np.zeros((n_m, m_p, p_n))
            for r in range(target_rank):
                T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            residual = np.max(np.abs(T - T_decomp))
            
            if residual < 1e-12:
                return U, V, W
            
            if residual > prev_residual * 1.1:
                break  # Diverging, try next initialization
            prev_residual = residual
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with trivial
    best_U, best_V, best_W = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But matrix multiplication rank is known to be less than n*m*p for some cases
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        result = _find_lower_rank(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                # Continue trying lower ranks
                continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
