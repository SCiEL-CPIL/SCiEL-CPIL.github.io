import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-4):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    target = _build_tensor(n, m, p)
    return np.allclose(T, target, atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_rank_decomposition(n, m, p, rank, seed, max_iters=5000, lr=0.01, deadline=None):
    """Try to find a rank-rank decomposition via gradient descent."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    
    # Initialize with small random values
    U = rng.randn(nm, rank) * 0.1
    V = rng.randn(mp, rank) * 0.1
    W = rng.randn(pn, rank) * 0.1
    
    # Precompute target tensor in flattened form for faster computation
    # T_flat has shape (nm, mp, pn)
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Compute current tensor
        T_cur = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_cur += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        
        # Compute loss
        diff = T_cur - T
        loss = np.sum(diff**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-8:
            break
        
        # Compute gradients
        # dL/dU[i,r] = 2 * sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
        # dL/dV[j,r] = 2 * sum_{i,k} diff[i,j,k] * U[i,r] * W[k,r]
        # dL/dW[k,r] = 2 * sum_{i,j} diff[i,j,k] * U[i,r] * V[j,r]
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(rank):
            # For each r, we have a rank-1 contribution
            # diff contribution from term r: U[:,r] * V[:,r] * W[:,r]
            # dL/dU[:,r] = 2 * sum_{j,k} diff[:,:,] * V[j,r]*W[k,r] for each i
            # This is: 2 * (diff contracted with V[:,r] over j and W[:,r] over k)
            # dU[i,r] = 2 * sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
            dU[:,r] = 2.0 * np.einsum('ijk,j,k->i', diff, V[:,r], W[:,r])
            dV[:,r] = 2.0 * np.einsum('ijk,i,k->j', diff, U[:,r], W[:,r])
            dW[:,r] = 2.0 * np.einsum('ijk,i,j->k', diff, U[:,r], V[:,r])
        
        # Gradient descent with learning rate decay
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        if it % 500 == 0 and it > 0:
            lr *= 0.5
    
    return best_U, best_V, best_W, best_loss

def _try_strassen_based(n, m, p):
    """Try to construct a decomposition using Strassen-like ideas for sub-blocks.
    
    For 4x5 x 5x5, we can try to view this as blocks and use low-rank 
    decompositions for the block multiplications.
    """
    # Try a block approach: split the 4x5 matrix into blocks
    # This is complex; let's try a simpler structured approach
    
    # Actually, let's try to use the known fact that 2x2x2 has rank 7.
    # We can pad the 4x5x5 problem to use multiple 2x2x2 blocks.
    # But the dimensions don't divide evenly.
    
    # Alternative: try rank 50 (half of schoolbook) with a structured initialization
    # based on pairing up terms
    
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    
    # Try pairing: for each (i,k), we need sum_j A[i,j]*B[j,k] = C[i,k]
    # The schoolbook has one term per (i,j,k). 
    # Can we combine terms with the same (i,k) but different j?
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # This is already a sum - the tensor is T[i*m+j, j*p+k, k*n+i] = 1
    # So for fixed (i,k), the terms are over j=0..m-1
    # Each term is: e_{i*m+j} * e_{j*p+k} * e_{k*n+i}
    # These are all different rank-1 tensors since they have different (i,j) or (j,k) components
    # So we can't simply combine them.
    
    # The tensor is inherently a sum of n*m*p rank-1 terms in the standard basis.
    # To get lower rank, we need to find a different decomposition.
    
    # Let's try a numerical approach with various ranks
    return None  # Fall through to numerical search

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    U_sch, V_sch, W_sch = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    best_rank = n * m * p
    
    # Try various ranks below schoolbook
    # The lower bound is max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    # Try ranks: 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95
    
    candidate_ranks = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds for this rank
        max_seeds = 3
        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                break
            
            U, V, W, loss = _try_rank_decomposition(
                n, m, p, rank, seed=seed + rank, 
                max_iters=3000, lr=0.05, deadline=deadline
            )
            
            if loss < 1e-6:
                # Validate
                if _validate(U, V, W, n, m, p, tol=1e-4):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break  # Found valid decomposition for this rank, try lower
            elif loss < 1e-3:
                # Close but not quite - try more iterations
                U, V, W, loss = _try_rank_decomposition(
                    n, m, p, rank, seed=seed + rank + 100,
                    max_iters=5000, lr=0.01, deadline=deadline
                )
                if loss < 1e-6:
                    if _validate(U, V, W, n, m, p, tol=1e-4):
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        break
    
    # If we found a better rank, use it; otherwise fall back to schoolbook
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}

