import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    def check_decomposition(U, V, W, atol=1e-6):
        # Compute tensor product
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)

    # Candidate values for discrete search
    # Rank 18 decompositions for 2x3x4 often use integers
    candidate_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    def try_rank(r, time_limit=10):
        """
        Attempts to find a decomposition of rank r using discrete search 
        and local refinement.
        """
        start = time.time()
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # We will perform a series of random restarts with local search
        num_restarts = 500
        
        for restart in range(num_restarts):
            if time.time() > start + time_limit:
                break
            
            # Initialize with random discrete values
            U = np.random.choice(candidate_values, size=(nm, r))
            V = np.random.choice(candidate_values, size=(mp, r))
            W = np.random.choice(candidate_values, size=(pn, r))
            
            # Local search: Hill Climbing
            improved = True
            while improved:
                if time.time() > start + time_limit:
                    break
                improved = False
                # Try flipping one entry at a time in U, V, W
                # To speed up, we only check a subset of indices per pass
                # or use a more efficient error update if possible.
                # For now, full recompute is safe but slow.
                
                # Shuffle indices to avoid bias
                idx_U = np.random.permutation(nm * r)
                for idx in idx_U[:100]: # Limit checks per iteration
                    i, j = divmod(idx, r)
                    old_val = U[i, j]
                    for new_val in candidate_values:
                        if new_val == old_val:
                            continue
                        U[i, j] = new_val
                        err = compute_error(U, V, W)
                        if err < best_err:
                            best_err = err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            improved = True
                        else:
                            U[i, j] = old_val
                    if improved:
                        break
                
                if not improved:
                    idx_V = np.random.permutation(mp * r)
                    for idx in idx_V[:100]:
                        i, j = divmod(idx, r)
                        old_val = V[i, j]
                        for new_val in candidate_values:
                            if new_val == old_val:
                                continue
                            V[i, j] = new_val
                            err = compute_error(U, V, W)
                            if err < best_err:
                                best_err = err
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                improved = True
                            else:
                                V[i, j] = old_val
                        if improved:
                            break
                
                if not improved:
                    idx_W = np.random.permutation(pn * r)
                    for idx in idx_W[:100]:
                        i, j = divmod(idx, r)
                        old_val = W[i, j]
                        for new_val in candidate_values:
                            if new_val == old_val:
                                continue
                            W[i, j] = new_val
                            err = compute_error(U, V, W)
                            if err < best_err:
                                best_err = err
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                improved = True
                            else:
                                W[i, j] = old_val
                        if improved:
                            break
                            
            if best_err < 1e-6:
                return best_U, best_V, best_W
                
        return None

    # Try ranks from low to high
    # Rank 18 is the known minimum for 2x3x4
    for r in [18, 19, 20]:
        if time.time() > deadline - 20:
            break
        res = try_rank(r, time_limit=15)
        if res is not None:
            U, V, W = res
            if check_decomposition(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
