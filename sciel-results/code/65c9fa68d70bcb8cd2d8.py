"""Matrix multiplication 2x4 * 4x5 tensor decomposition - rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # leave 10s margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 40
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is valid using einsum."""
        r = U.shape[1]
        U_idx = U.reshape(n, m, r)
        V_idx = V.reshape(m, p, r)
        W_idx = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def als_decompose(target_rank, max_iters=200, tol=1e-10, seed=42):
        """Alternating Least Squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        prev_err = float('inf')
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
                
            # Fix V and W, solve for U
            for i in range(n):
                for j in range(m):
                    # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve U[i*m+j, :] * M.T = ones(p)
                    # i.e., M.T @ U[i*m+j, :].T = ones(p)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            U_idx = U.reshape(n, m, target_rank)
            V_idx = V.reshape(m, p, target_rank)
            W_idx = W.reshape(p, n, target_rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
            err = np.max(np.abs(T - 1.0))
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W, prev_err
    
    # Try ranks from lower bound up
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try various target ranks with multiple random seeds
    for target_rank in range(20, rank_school + 1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            
            U, V, W, err = als_decompose(target_rank, max_iters=300, seed=seed)
            
            if err < 1e-6:
                # Found a valid decomposition
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            # Try to reduce further
            continue
        else:
            # If we can't find a decomposition at this rank, 
            # try a few more seeds before giving up
            pass
    
    # Try to round to integer/half-integer values for cleaner solution
    if best_rank < rank_school:
        # Round to nearest 0.5
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        
        if verify(U_round, V_round, W_round, tol=1e-4):
            best_U = U_round
            best_V = V_round
            best_W = W_round
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
