"""Matrix multiplication 2,3,9 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    # Build the tensor T of shape (n, m, p)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a 3-tensor product.
    # Efficient way: for each c, the outer product of U[:,c], V[:,c], W[:,c]
    # But we can do it index-wise.
    
    # Reshape for einsum:
    # U_flat[i*m+j] -> U2[i, j]
    # V_flat[j*p+k] -> V2[j, k]
    # W_flat[k*n+i] -> W2[k, i]
    
    U2 = U.reshape(n, m, -1)  # (n, m, rank)
    V2 = V.reshape(m, p, -1)  # (m, p, rank)
    W2 = W.reshape(p, n, -1)  # (p, n, rank)
    
    # T[i,j,k] = sum_c U2[i,j,c] * V2[j,k,c] * W2[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U2, V2, W2)
    
    # Check if all entries are 1 (within tolerance for float, or exact for int)
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    
    # Also check integrality if we claim integer/half-integer
    # For now, just check the equality
    return True

def _try_rank_reduction(n, m, p, max_rank, seed=42, time_limit=30):
    """
    Try to find a decomposition of rank < n*m*p using random initialization + gradient descent.
    This is a heuristic approach. If it finds a valid decomposition, return it.
    Otherwise, return None.
    """
    np.random.seed(seed)
    start = time.time()
    
    # We try to minimize the loss: sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    # Parameters: U (n*m, r), V (m*p, r), W (p*n, r)
    # Total parameters: (n*m + m*p + p*n) * r
    
    best_rank = max_rank
    best_params = None
    
    # Try a few different ranks
    for r in range(20, max_rank + 1):  # Try ranks from 20 up to max_rank
        if time.time() - start > time_limit:
            break
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Simple gradient descent
        lr = 0.01
        for step in range(1000):
            if time.time() - start > time_limit:
                break
            
            # Compute T
            U2 = U.reshape(n, m, r)
            V2 = V.reshape(m, p, r)
            W2 = W.reshape(p, n, r)
            T = np.einsum('ijc,jkc,kic->ijk', U2, V2, W2)
            
            # Loss
            err = T - np.ones((n, m, p))
            loss = np.sum(err ** 2)
            
            if loss < 1e-12:
                # Verify exactly
                if _verify(U, V, W, n, m, p):
                    best_rank = r
                    best_params = (U, V, W)
                    break
            
            # Compute gradients
            # dL/dU[i*m+j,c] = 2 * err[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
            # Sum over k
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # This is expensive to do in a loop, but for small dimensions it's okay
            # Or use autograd-like approach with einsum
            # dU[i,j,c] = 2 * sum_k err[i,j,k] * V2[j,k,c] * W2[k,i,c]
            # dV[j,k,c] = 2 * sum_i err[i,j,k] * U2[i,j,c] * W2[k,i,c]
            # dW[k,i,c] = 2 * sum_j err[i,j,k] * U2[i,j,c] * V2[j,k,c]
            
            # Use einsum for efficiency
            # dU2 = 2 * einsum('ijk,jkc,kic->ijc', err, V2, W2)
            # dV2 = 2 * einsum('ijk,ijc,kic->jkc', err, U2, W2)
            # dW2 = 2 * einsum('ijk,ijc,jkc->kic', err, U2, V2)
            
            dU2 = 2 * np.einsum('ijk,jkc,kic->ijc', err, V2, W2)
            dV2 = 2 * np.einsum('ijk,ijc,kic->jkc', err, U2, W2)
            dW2 = 2 * np.einsum('ijk,ijc,jkc->kic', err, U2, V2)
            
            dU = dU2.reshape(n * m, r)
            dV = dV2.reshape(m * p, r)
            dW = dW2.reshape(p * n, r)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if we found a valid solution
        U2 = U.reshape(n, m, r)
        V2 = V.reshape(m, p, r)
        W2 = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U2, V2, W2)
        if np.allclose(T, np.ones((n, m, p)), atol=1e-9):
            if _verify(U, V, W, n, m, p):
                best_rank = r
                best_params = (U, V, W)
                break
    
    if best_params is not None:
        return best_params[0], best_params[1], best_params[2]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    current_rank = schoolbook_rank
    
    # Try to reduce rank using optimization
    # We have about 100 seconds of budget.
    # Try rank 36, 42, 48, etc.
    # Actually, let's just try to find *any* rank < 54.
    
    # Try a few seeds and rank targets
    seeds = [42, 123, 456]
    time_per_seed = 20  # seconds
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_reduction(n, m, p, schoolbook_rank, seed=seed, time_limit=time_per_seed)
        if result is not None:
            U, V, W = result
            current_rank = U.shape[1]
            # If we found a lower rank, we can stop early if it's very good
            if current_rank <= 36:
                break
    
    # Final check
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        current_rank = schoolbook_rank
    
    # Convert to lists of lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
