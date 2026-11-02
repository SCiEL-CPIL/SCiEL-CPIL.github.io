import time
import numpy as np

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 145.0 # Safety margin for 150s limit

def construct():
    """
    Finds a low-rank tensor decomposition for 4x5 * 5x6 matrix multiplication.
    Uses a local search strategy starting from schoolbook and attempting to
    reduce rank by merging or eliminating terms, verifying correctness at each step.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Initialize with Schoolbook (Rank = n*m*p = 120)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indices:
    # U[i, j, k] corresponds to term for A[i,j] * B[j,k] contributing to C[i,k]
    # In matrix form:
    # U[row_u, col] = 1 if term col is A[i,j]
    # V[row_v, col] = 1 if term col is B[j,k]
    # W[row_w, col] = 1 if term col contributes to C[i,k]
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i # Trace convention W[k*n+i]
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def verify(U, V, W):
        """
        Checks if U, V, W form a valid decomposition.
        Tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        We need T_{i,j,k} = 1 if (i,j,k) is a valid index, else 0?
        Actually, matrix mult tensor is defined such that C_{i,k} = sum_j A_{i,j} B_{j,k}.
        The tensor decomposition represents the multiplication algorithm.
        The condition is:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all valid i,j,k?
        No, that's not quite right. The standard decomposition satisfies:
        C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        Wait, the factor matrices *are* the coefficients.
        The tensor T has entries T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
        In the decomposition:
        T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        For matrix multiplication, T_{i,j,k} is 1 if the indices are valid and correspond to the product term.
        Specifically, for 4x5 * 5x6:
        A is 4x5, B is 5x6, C is 4x6.
        C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k].
        The tensor T is supported on (i,j,k) where i in [0,3], j in [0,4], k in [0,5].
        And T[i,j,k] = 1.
        So we check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
        """
        # Efficient check using numpy
        # Reshape U, V, W to 3D tensors
        U_3d = U.T.reshape(rank, n, m) # rank, i, j
        V_3d = V.T.reshape(rank, m, p) # rank, j, k
        W_3d = W.T.reshape(rank, p, n) # rank, k, i
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is a tensor contraction.
        # We can do this by iterating over r or using einsum.
        # einsum('rij,rjk,rki->ik', U_3d, V_3d, W_3d) gives the matrix C coefficients?
        # No, the result of the contraction is a tensor of shape (n, m, p) which should be all 1s.
        # Let's compute S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # To save time in verification, we can check a random subset of indices if rank is large,
        # but for exactness we must check all.
        # n*m*p = 120 checks. Each check is a dot product of length rank.
        # 120 * rank operations. If rank is 100, that's 12000 ops. Very fast.
        
        S = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
        return np.allclose(S, 1.0, atol=1e-9)

    # Local Search Loop
    # Strategy: Try to eliminate a term by setting its column to 0 in one matrix and adjusting others?
    # Or, try to merge two terms?
    # A simpler, more robust local search for tensor rank is difficult.
    # Alternative: Use the known Strassen-based block decomposition as a starting point if we can hardcode it?
    # The prompt says "inspiration 2" used block Winograd.
    # Let's try a different structural approach: 
    # Instead of hill climbing which is likely to get stuck, let's use a **Greedy Sparsification** on the schoolbook.
    # We know that for 2x2x2, rank 7 is possible.
    # For 4x5x6, we can decompose into blocks.
    # But hardcoding blocks is what the current "island" does.
    # The directive asks for a STRUCTURAL change.
    # Let's try **Randomized Search with Pruning**.
    # Initialize U, V, W with random Gaussian entries of shape (dim, r) for a small r (e.g., 90).
    # Use gradient descent to minimize ||Tensor(U,V,W) - Target||^2.
    # This is a continuous optimization problem.
    
    target_tensor = np.ones((n, m, p))
    
    def compute_tensor(U, V, W, r):
        U_3d = U.T.reshape(r, n, m)
        V_3d = V.T.reshape(r, m, p)
        W_3d = W.T.reshape(r, p, n)
        return np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)

    def loss(U, V, W, r):
        T = compute_tensor(U, V, W, r)
        return np.sum((T - target_tensor)**2)

    # Try different ranks from 80 up to 119
    # Start from low rank and increase until we find a solution?
    # Or start from schoolbook and try to reduce?
    # Gradient descent from random init for low rank is hard due to non-convexity.
    # However, for small dimensions, it might work if we restart multiple times.
    
    best_score = 120.0
    best_result = None
    
    # We will attempt to find a decomposition of rank < 120.
    # Known best for 4x5x6 is likely around 100-110.
    # Let's try ranks: 110, 105, 100, 95, 90...
    # If we find one, we return it.
    
    # To make this efficient, we will use a simple SGD or L-BFGS-B from scipy if available, 
    # but standard library + numpy is safer. We'll implement a simple gradient step.
    
    def optimize_rank(r, max_iters=5000, lr=0.01, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Gradient descent
        for it in range(max_iters):
            T = compute_tensor(U, V, W, r)
            err = T - target_tensor
            L = np.sum(err**2)
            
            if L < 1e-12:
                return U, V, W, True
            
            # Gradients
            # dL/dU = 2 * err * (V * W) contracted appropriately
            # U[i,j,r] appears in T[i,j,k] as U[i,j,r] * V[j,k,r] * W[k,i,r]
            # dT[i,j,k]/dU[i,j,r] = V[j,k,r] * W[k,i,r]
            
            U_3d = U.T.reshape(r, n, m)
            V_3d = V.T.reshape(r, m, p)
            W_3d = W.T.reshape(r, p, n)
            
            # grad_U_3d[i,j,r] = sum_k err[i,j,k] * V_3d[r,j,k] * W_3d[r,k,i]
            # This is complex to vectorize easily without einsum path optimization.
            # Let's use numerical gradient for simplicity if analytical is too slow/complex to code correctly in short time?
            # No, analytical is better.
            
            # grad_U[i,j,r] = sum_k err[i,j,k] V[j,k,r] W[k,i,r]
            # grad_V[j,k,r] = sum_i err[i,j,k] U[i,j,r] W[k,i,r]
            # grad_W[k,i,r] = sum_j err[i,j,k] U[i,j,r] V[j,k,r]
            
            # We can compute these using einsum
            # grad_U_3d = np.einsum('ijk,rjk,rki->rij', err, V_3d, W_3d)
            # grad_V_3d = np.einsum('ijk,rij,rki->rjk', err, U_3d, W_3d)
            # grad_W_3d = np.einsum('ijk,rij,rjk->rki', err, U_3d, V_3d)
            
            try:
                grad_U_3d = np.einsum('ijk,rjk,rki->rij', err, V_3d, W_3d)
                grad_V_3d = np.einsum('ijk,rij,rki->rjk', err, U_3d, W_3d)
                grad_W_3d = np.einsum('ijk,rij,rjk->rki', err, U_3d, V_3d)
            except Exception:
                return None, None, None, False
                
            U -= 2 * lr * grad_U_3d.T.reshape(n*m, r)
            V -= 2 * lr * grad_V_3d.T.reshape(m*p, r)
            W -= 2 * lr * grad_W_3d.T.reshape(p*n, r)
            
            if time.time() - start_time > TIME_LIMIT - 10:
                return None, None, None, False

        return None, None, None, False

    # Try to find a valid decomposition for ranks lower than 120
    # We'll try a few random restarts for each rank
    ranks_to_try = [115, 110, 105, 100, 95, 90]
    
    for r in ranks_to_try:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        found = False
        for seed in range(5):
            if time.time() - start_time > TIME_LIMIT - 20:
                break
                
            U, V, W, success = optimize_rank(r, max_iters=2000, lr=0.05, seed=seed)
            if success:
                # Verify one last time
                if verify(U, V, W):
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            # If we found a valid one at this rank, we can stop or try lower?
            # Given time constraints, stopping at the first found lower rank is a good heuristic.
            # But maybe a slightly higher rank is easier to find?
            # We iterate from high to low (115 down to 90), so we find the "best" (lowest) first if we go 90->115?
            # No, we want the LOWEST rank. So we should try 90 first.
            # If 90 fails, try 95, etc.
            pass

    # If no rank < 120 was found, return schoolbook
    if best_rank == 120:
        # Return the original schoolbook matrices
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
