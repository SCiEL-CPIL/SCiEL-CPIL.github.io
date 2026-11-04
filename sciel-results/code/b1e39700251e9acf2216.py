"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that U @ V^T @ W^T reconstructs the matmul tensor."""
    Ut = U.reshape(n, m, -1)
    Vt = V.reshape(m, p, -1)
    Wt = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)


def _rank(U, V, W):
    return U.shape[1]


def _compute_tensor(U, V, W, n, m, p):
    """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
    r = U.shape[1]
    Ut = U.reshape(n, m, r)
    Vt = V.reshape(m, p, r)
    Wt = W.reshape(p, n, r)
    return np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)


def _als_decompose(U_init, V_init, W_init, n, m, p, max_iter=200, tol=1e-10, time_limit=None):
    """
    Alternating Least Squares for tensor decomposition.
    For each factor, solve a linear system given the other two fixed.
    """
    U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
    r = U.shape[1]
    target = np.ones((n, m, p))
    deadline = time.time() + time_limit if time_limit else None
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 3:
            break
        
        # Update U: for each (i,j), solve U[i,j,:] such that
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
        # This is: U[i,j,:] @ A_{ij} = ones(p) where A_{ij}[k,c] = V[j,k,c]*W[k,i,c]
        for i in range(n):
            for j in range(m):
                # Build matrix A of shape (p, r) where A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A^T @ x = ones(p) where x = U[i*m+j, :]^T
                try:
                    if r >= p:
                        # Use lstsq for overdetermined or square
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                    else:
                        # Underdetermined: use pinv
                        sol = np.linalg.pinv(A.T) @ np.ones(p)
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k), solve V[j,k,:] such that
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i
        for j in range(m):
            for k in range(p):
                # Build matrix B of shape (n, r) where B[i,c] = U[i*m+j,c] * W[k*n+i,c]
                B = np.zeros((n, r))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    if r >= n:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                    else:
                        sol = np.linalg.pinv(B.T) @ np.ones(n)
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i), solve W[k,i,:] such that
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
        for k in range(p):
            for i in range(n):
                # Build matrix C of shape (m, r) where C[j,c] = U[i*m+j,c] * V[j*p+k,c]
                C = np.zeros((m, r))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    if r >= m:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(C.T, np.ones(m), rcond=None)
                    else:
                        sol = np.linalg.pinv(C.T) @ np.ones(m)
                    W[k*n+i, :] = sol
                except:
                    pass
        
        # Check convergence
        T = _compute_tensor(U, V, W, n, m, p)
        err = np.sum((T - target)**2)
        if err < tol:
            break
    
    T = _compute_tensor(U, V, W, n, m, p)
    if np.allclose(T, target, atol=1e-6):
        return U, V, W
    return None


def _try_merge(U, V, W, n, m, p, time_limit=None):
    """
    Try to reduce rank by 1 by finding two columns that can be merged.
    For each pair (c1, c2), try to find a single column that represents
    the sum of their contributions. This works if the two terms have
    overlapping structure.
    """
    r = U.shape[1]
    target = np.ones((n, m, p))
    deadline = time.time() + time_limit if time_limit else None
    
    # Try all pairs of columns to merge
    best_result = None
    best_err = float('inf')
    
    # Limit the number of pairs we try based on time
    max_pairs = 50
    pair_count = 0
    
    for c1 in range(r):
        if deadline and time.time() > deadline - 5:
            break
        if pair_count >= max_pairs:
            break
        for c2 in range(c1 + 1, r):
            if deadline and time.time() > deadline - 5:
                break
            if pair_count >= max_pairs:
                break
            pair_count += 1
            
            # Remove columns c1 and c2, add a new column
            # The new column should approximate U[:,c1]+U[:,c2], etc.
            # But we need to solve for the new column exactly.
            
            # New rank is r-1
            new_r = r - 1
            new_U = np.delete(U, [c1, c2], axis=1)
            new_V = np.delete(V, [c1, c2], axis=1)
            new_W = np.delete(W, [c1, c2], axis=1)
            
            # Add a new column at the end
            new_U = np.column_stack([new_U, np.zeros(n*m)])
            new_V = np.column_stack([new_V, np.zeros(m*p)])
            new_W = np.column_stack([new_W, np.zeros(p*n)])
            
            # Try multiple initializations for the new column
            inits = [
                U[:, c1] + U[:, c2],
                U[:, c1] - U[:, c2],
                U[:, c2] - U[:, c1],
            ]
            vinits = [
                V[:, c1] + V[:, c2],
                V[:, c1] - V[:, c2],
                V[:, c2] - V[:, c1],
            ]
            winits = [
                W[:, c1] + W[:, c2],
                W[:, c1] - W[:, c2],
                W[:, c2] - W[:, c1],
            ]
            
            for init_idx in range(3):
                if deadline and time.time() > deadline - 2:
                    break
                
                new_U[:, new_r-1] = inits[init_idx]
                new_V[:, new_r-1] = vinits[init_idx]
                new_W[:, new_r-1] = winits[init_idx]
                
                # Polish with ALS
                result = _als_decompose(new_U, new_V, new_W, n, m, p, 
                                         max_iter=50, tol=1e-10, time_limit=1)
                if result is not None:
                    return result
                # Also try without polishing if already close
                T = _compute_tensor(new_U, new_V, new_W, n, m, p)
                err = np.sum((T - target)**2)
                if err < 1e-6:
                    return new_U, new_V, new_W
                if err < best_err:
                    best_err = err
                    best_result = (new_U.copy(), new_V.copy(), new_W.copy())
    
    if best_result is not None and best_err < 1e-4:
        # Try to polish the best result
        U_b, V_b, W_b = best_result
        result = _als_decompose(U_b, V_b, W_b, n, m, p, max_iter=100, tol=1e-10, time_limit=3)
        if result is not None:
            return result
    
    return None


def _random_init(n, m, p, r, seed=0):
    """Generate a random initialization for rank-r decomposition."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)
    # Normalize
    for c in range(r):
        norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
        if norm > 1e-10:
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
    return U, V, W


def _try_rank_random(r, n, m, p, n_restarts=5, time_limit=None):
    """Try to find a rank-r decomposition using random init + ALS."""
    deadline = time.time() + time_limit if time_limit else None
    for seed in range(n_restarts):
        if deadline and time.time() > deadline - 10:
            break
        U, V, W = _random_init(n, m, p, r, seed=seed)
        result = _als_decompose(U, V, W, n, m, p, max_iter=300, tol=1e-10, time_limit=5)
        if result is not None:
            return result
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = _rank(U, V, W)
    best_result = (U, V, W)
    
    # Strategy 1: Iterative merge from schoolbook
    current_U, current_V, current_W = U, V, W
    current_rank = best_rank
    merge_count = 0
    while time.time() < deadline - 30 and merge_count < 20:
        remaining = deadline - 30 - time.time()
        if remaining < 2:
            break
        result = _try_merge(current_U, current_V, current_W, n, m, p, time_limit=min(remaining * 0.3, 3))
        if result is not None:
            current_U, current_V, current_W = result
            current_rank = _rank(current_U, current_V, current_W)
            merge_count += 1
            if current_rank < best_rank:
                best_rank = current_rank
                best_result = (current_U.copy(), current_V.copy(), current_W.copy())
        else:
            break
    
    # Strategy 2: Try random init + ALS for ranks lower than current best
    for r in range(best_rank - 1, max(10, best_rank // 2), -1):
        if time.time() > deadline - 15:
            break
        time_per_attempt = max(1, (deadline - 15 - time.time()) / 3)
        result = _try_rank_random(r, n, m, p, n_restarts=5, time_limit=time_per_attempt)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # Final polish of best result
    U, V, W = best_result
    if time.time() < deadline - 5:
        polished = _als_decompose(U, V, W, n, m, p, max_iter=200, tol=1e-12, time_limit=5)
        if polished is not None:
            U, V, W = polished
    
    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-5):
        # Fall back to schoolbook
        U, V, W = _schoolbook()
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
