import numpy as np
import time
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115  # Safety margin
    n, m, p = 4, 5, 6
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Lower bound for rank
    lower_bound = max(n * p, m * p, n * m)
    
    # We will try a few ranks near the lower bound.
    # If we don't find a better one quickly, we stick with schoolbook.
    # For 4x5x6, lower_bound=30.
    # Let's try ranks 30 to 35 with a very fast method.
    
    for rank in range(lower_bound, min(lower_bound + 10, best_rank)):
        if time.time() > deadline:
            break
        result = _try_rank_fast(n, m, p, rank, deadline)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                # Since we are iterating upwards, if we find one, we can stop 
                # or continue to find lower? No, we start from lower bound.
                # So the first one found is the lowest among those tried.
                # But we might have skipped some? No, we iterate up.
                # So if we find rank 30, we can stop.
                # If we don't find 30, we try 31, etc.
                break
                
    return {"U": np.array(best_U), "V": np.array(best_V), "W": np.array(best_W)}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank_fast(n, m, p, rank, deadline):
    """
    Fast attempt to find rank-r decomposition.
    Uses random initialization and a few steps of gradient descent with vectorized updates.
    Returns None if not found within time/iterations.
    """
    if time.time() > deadline:
        return None
        
    np.random.seed(42)
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Pre-compute indices for valid entries
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_a.append(i * m + j)
                indices_b.append(j * p + k)
                indices_c.append(k * n + i)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    # Targets are all 1.0
    targets = np.ones(len(indices_a))
    
    # Also include some zero entries to prevent trivial solutions?
    # The tensor is 0 for invalid indices.
    # Adding zero constraints helps constrain the solution.
    # Let's add 50 random invalid entries.
    np.random.seed(123)
    invalid_a = []
    invalid_b = []
    invalid_c = []
    for _ in range(50):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        # Check if valid
        i1, j1 = divmod(a, m)
        j2, k2 = divmod(b, p)
        k1, i2 = divmod(c, n)
        if not (i1 == i2 and j1 == j2 and k2 == k1):
            invalid_a.append(a)
            invalid_b.append(b)
            invalid_c.append(c)
            
    if invalid_a:
        indices_a = np.concatenate([indices_a, invalid_a])
        indices_b = np.concatenate([indices_b, invalid_b])
        indices_c = np.concatenate([indices_c, invalid_c])
        targets = np.concatenate([targets, np.zeros(len(invalid_a))])
        
    lr = 0.05
    max_iters = 200
    
    for it in range(max_iters):
        if time.time() > deadline:
            return None
            
        # Compute current values: U[a] dot (V[b] * W[c])
        # Vectorized: 
        # U[indices_a] is (N, R)
        # V[indices_b] is (N, R)
        # W[indices_c] is (N, R)
        # We need sum_r U[a,r] V[b,r] W[c,r]
        
        U_sel = U[indices_a]
        V_sel = V[indices_b]
        W_sel = W[indices_c]
        
        # Element-wise product and sum over rank
        current = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        errors = current - targets
        
        # Check convergence
        mse = np.mean(errors ** 2)
        if mse < 1e-10:
            # Verify exactly
            if _verify_decomposition_fast(n, m, p, U, V, W):
                return U, V, W
            # If not exact, continue or break?
            # If it's very small, it might be a numerical issue.
            # Let's try to polish.
            
        # Compute gradients
        # dU[a,r] = sum_{(a,b,c) where a is fixed} error * V[b,r] * W[c,r]
        # This is tricky to vectorize fully without sparse structures because 
        # multiple (b,c) pairs can share the same 'a'.
        # However, N is small (120 + 50 = 170).
        # We can just loop over the N entries for gradient accumulation, 
        # but use vectorized operations for the rank dimension.
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Loop over entries (N is ~170, this is fast)
        for idx in range(len(indices_a)):
            a = indices_a[idx]
            b = indices_b[idx]
            c = indices_c[idx]
            e = errors[idx]
            
            # Vectorized update for this entry
            dU[a] += e * V_sel[idx] * W_sel[idx]
            dV[b] += e * U_sel[idx] * W_sel[idx]
            dW[c] += e * U_sel[idx] * V_sel[idx]
            
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
    # Final check
    if _verify_decomposition_fast(n, m, p, U, V, W):
        return U, V, W
        
    return None

def _verify_decomposition_fast(n, m, p, U, V, W, tol=1e-8):
    """Fast verification using NumPy."""
    # Check all valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
                    
    # Check a few invalid entries
    np.random.seed(42)
    for _ in range(20):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        i1, j1 = divmod(a, m)
        j2, k2 = divmod(b, p)
        k1, i2 = divmod(c, n)
        if not (i1 == i2 and j1 == j2 and k2 == k1):
            val = np.dot(U[a], V[b] * W[c])
            if abs(val) > tol:
                return False
                
    return True
# EVOLVE-BLOCK-END
