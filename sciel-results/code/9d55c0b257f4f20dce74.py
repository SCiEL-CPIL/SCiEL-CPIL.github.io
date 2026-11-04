# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    
    # Schoolbook indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    for idx, (a, b, c) in enumerate(indices):
        U_fb[a, idx] = 1
        V_fb[b, idx] = 1
        W_fb[c, idx] = 1
        
    # Precompute the target tensor as a sparse structure or dense array
    # Dense array of size (n*m, m*p, p*n) is 12*24*18 = 5184 elements. Very small.
    T = np.zeros((n * m, m * p, p * n))
    for a, b, c in indices:
        T[a, b, c] = 1.0

    best_result = None
    best_rank = rank_schoolbook

    # Try to find a lower rank
    # Lower bound: max(n*m, m*p, p*n) = 24
    # Try ranks from 24 to 71
    for target_rank in range(24, rank_schoolbook):
        if time.time() > deadline - 10:
            break
            
        # Use a few random restarts
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(attempt * 100 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple Gradient Descent with vectorized error
            lr = 0.01
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                    
                # Compute current decomposition
                # Sum_r U[:, r] * V[:, r] * W[:, r]
                # This is a tensor of shape (n*m, m*p, p*n)
                # Can be computed as:
                # for r in range(target_rank):
                #   T_est += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                # But we can do it faster:
                # T_est = np.einsum('ir, jr, kr -> ijk', U, V, W)
                
                T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
                error = np.sum((T_est - T)**2)
                
                if error < 1e-8:
                    # Found a solution
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    break
                
                # Compute gradients
                # dE/dU = 2 * sum_{j,k} (T_est - T) * V[j,r] * W[k,r]
                # This is also vectorizable
                diff = T_est - T
                # grad_U[i, r] = sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
                # grad_U = np.einsum('ijk,jr,kr->ir', diff, V, W)
                grad_U = np.einsum('ijk,jr,kr->ir', diff, V, W)
                grad_V = np.einsum('ijk,ir,kr->jr', diff, U, W)
                grad_W = np.einsum('ijk,ir,jr->kr', diff, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            if best_result:
                break
        
        if best_result:
            break
            
    if best_result:
        return best_result
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
