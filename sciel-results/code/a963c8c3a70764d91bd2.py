import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W, rank


def _compute_tensor_fast(U, V, W, n, m, p):
    """Compute the tensor from the decomposition using vectorized operations."""
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    # U is (n*m, r), row i*m+j corresponds to (i,j). Reshape preserves this order.
    U_reshaped = U.reshape(n, m, -1) 
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Note: W index is k*n+i, so W[k, i, c] in reshaped form (p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return T


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is valid."""
    T_decomp = _compute_tensor_fast(U, V, W, n, m, p)
    T_target = np.ones((n, m, p))
    return np.allclose(T_decomp, T_target, atol=tol)


def _try_local_search(U, V, W, n, m, p, max_iter=500, tol=1e-6, deadline=None):
    """Try to refine the decomposition using coordinate descent with fast tensor eval."""
    rank = U.shape[1]
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    T_target = np.ones((n, m, p))
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        T_decomp = _compute_tensor_fast(U, V, W, n, m, p)
        error = np.max(np.abs(T_decomp - T_target))
        
        if error < tol:
            return U, V, W, True
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try adjusting each term
        improved = False
        for c in range(rank):
            # Try small perturbations to U[:,c]
            for delta in [-0.5, 0.5]:
                U_old = U[:, c].copy()
                U[:, c] += delta
                T_new = _compute_tensor_fast(U, V, W, n, m, p)
                err_new = np.max(np.abs(T_new - T_target))
                if err_new < error:
                    error = err_new
                    improved = True
                else:
                    U[:, c] = U_old
            
            # Try small perturbations to V[:,c]
            for delta in [-0.5, 0.5]:
                V_old = V[:, c].copy()
                V[:, c] += delta
                T_new = _compute_tensor_fast(U, V, W, n, m, p)
                err_new = np.max(np.abs(T_new - T_target))
                if err_new < error:
                    error = err_new
                    improved = True
                else:
                    V[:, c] = V_old
            
            # Try small perturbations to W[:,c]
            for delta in [-0.5, 0.5]:
                W_old = W[:, c].copy()
                W[:, c] += delta
                T_new = _compute_tensor_fast(U, V, W, n, m, p)
                err_new = np.max(np.abs(T_new - T_target))
                if err_new < error:
                    error = err_new
                    improved = True
                else:
                    W[:, c] = W_old
        
        if not improved:
            break
    
    # Check if we found a valid solution
    T_final = _compute_tensor_fast(U, V, W, n, m, p)
    if np.allclose(T_final, T_target, atol=tol):
        return U, V, W, True
    return best_U, best_V, best_W, False


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), rank_sb
    
    np.random.seed(42)
    
    # Try to find lower rank decompositions
    # Known ranks for 3x3x5 are likely < 30. Let's try 15 to 35.
    for target_rank in range(15, 36):
        if time.time() > deadline - 10:
            break
        found = False
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Random sparse initialization
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for c in range(target_rank):
                # Random non-zero pattern
                # Use denser initialization to help convergence
                n_nonzero_U = np.random.randint(3, 7)
                n_nonzero_V = np.random.randint(3, 7)
                n_nonzero_W = np.random.randint(3, 7)
                
                idx_U = np.random.choice(n*m, size=n_nonzero_U, replace=False)
                idx_V = np.random.choice(m*p, size=n_nonzero_V, replace=False)
                idx_W = np.random.choice(p*n, size=n_nonzero_W, replace=False)
                
                for idx in idx_U:
                    U[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                for idx in idx_V:
                    V[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                for idx in idx_W:
                    W[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # Try local search
            U_ls, V_ls, W_ls, success = _try_local_search(U, V, W, n, m, p, max_iter=200, deadline=deadline - 10)
            
            if success:
                best_U, best_V, best_W, best_rank = U_ls, V_ls, W_ls, target_rank
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
