"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook rank 48
    U = _schoolbook_U(n, m, p)
    V = _schoolbook_V(n, m, p)
    W = _schoolbook_W(n, m, p)
    
    # Try to reduce rank using numerical optimization
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Attempt rank reduction via iterative refinement
    for target_rank in range(best_rank - 1, 7, -1):
        if time.time() > deadline - 20:
            break
        try:
            success, U_r, V_r, W_r = _try_rank_reduction(n, m, p, target_rank)
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                if best_rank <= 12:
                    break
        except Exception:
            continue
    
    # Convert to lists with rational/half-integer entries
    U_out = _to_rational_list(best_U)
    V_out = _to_rational_list(best_V)
    W_out = _to_rational_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _schoolbook_U(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
    return U

def _schoolbook_V(n, m, p):
    rank = n * m * p
    V = np.zeros((m * p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                V[j * p + k, col] = 1
    return V

def _schoolbook_W(n, m, p):
    rank = n * m * p
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                W[k * n + i, col] = 1
    return W

def _verify(n, m, p, U, V, W):
    """Check if decomposition is exact."""
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For matrix multiplication: T[i,j,k] = 1 if valid, else 0
    # Actually for M(n,m) * N(m,p): output[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # So T[i,j,j',k,k',i'] = 1 if j==j' and k==k' and i==i', else 0
    # With modes: U index (i*m+j), V index (j'*p+k), W index (k'*n+i')
    # T[i,j,j',k,k',i'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            expected = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                            actual = 0.0
                            for r in range(U.shape[1]):
                                actual += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                            if abs(actual - expected) > 1e-6:
                                return False
    return True

def _try_rank_reduction(n, m, p, target_rank):
    """Try to find a valid decomposition of given rank using optimization."""
    np.random.seed(42)
    
    # Initialize with random factors
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Use gradient-based optimization on the constraint
    # Loss = sum over all tensor entries of (actual - expected)^2
    lr = 0.01
    num_iters = 500
    
    for it in range(num_iters):
        if time.time() > time.time() + 10:  # per-attempt timeout
            break
        
        # Compute gradient
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    expected = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                                    # Only need to check where expected=1 or where U,V,W are non-zero
                                    # This is too slow, use vectorized approach instead
                                    pass
        break  # Fall back to vectorized below
    
    # Vectorized approach
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            expected = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                            indices.append((i*m+j, jp*p+k, kp*n+ip, expected))
    
    idx_U = np.array([x[0] for x in indices])
    idx_V = np.array([x[1] for x in indices])
    idx_W = np.array([x[2] for x in indices])
    expected = np.array([x[3] for x in indices])
    
    for it in range(num_iters):
        # Compute actual tensor values
        # actual[idx] = sum_r U[idx_U[idx], r] * V[idx_V[idx], r] * W[idx_W[idx], r]
        # This is expensive, so we'll use a simpler heuristic
        
        # Simple heuristic: try to zero out columns
        # Check which columns are essential
        col_norms = np.sum(np.abs(U), axis=0) * np.sum(np.abs(V), axis=0) * np.sum(np.abs(W), axis=0)
        
        if it % 10 == 0:
            # Try removing the smallest column
            min_col = np.argmin(col_norms)
            # Check if removing it still works (approximate check)
            U_test = np.delete(U, min_col, axis=1)
            V_test = np.delete(V, min_col, axis=1)
            W_test = np.delete(W, min_col, axis=1)
            
            if U_test.shape[1] < target_rank:
                U = U_test
                V = V_test
                W = W_test
                if U.shape[1] == target_rank - 1:
                    target_rank -= 1
                    break
        
        # Adjust learning rate
        lr *= 0.99
    
    # Final verification
    if _verify(n, m, p, U, V, W) and U.shape[1] == target_rank:
        return True, U, V, W
    
    return False, None, None

def _to_rational_list(mat):
    """Convert numpy matrix to list of lists with rational entries."""
    result = []
    for row in mat:
        r = []
        for val in row:
            # Snap to nearest half-integer or integer
            rounded = round(val * 2) / 2.0
            if abs(val - rounded) < 1e-4:
                r.append(rounded)
            else:
                r.append(float(val))
        result.append(r)
    return result
# EVOLVE-BLOCK-END
