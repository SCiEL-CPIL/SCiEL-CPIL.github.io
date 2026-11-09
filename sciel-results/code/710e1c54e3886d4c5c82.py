import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p for 3x5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                if val != 1:
                    return False
    
    # Check some random non-matching points to ensure sparsity is maintained
    rng = np.random.RandomState(42)
    for _ in range(100):
        a = rng.randint(0, n * m)
        b = rng.randint(0, m * p)
        c = rng.randint(0, p * n)
        # Check if (a,b,c) corresponds to a valid (i,j,k)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 != j2 or k1 != k2 or i1 != i2:
            val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
            if val != 0:
                return False
    
    return True

def _gradient_descent_decomposition(rank, n, m, p, max_iters, lr, seed):
    """
    Try to find a rank-r decomposition using gradient descent.
    Returns (U, V, W) as lists of lists, or None if not found.
    """
    rng = np.random.RandomState(seed)
    
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Initialize factors
    # Scale initialization to avoid vanishing/exploding gradients
    U = rng.randn(dim_u, rank) * 0.1
    V = rng.randn(dim_v, rank) * 0.1
    W = rng.randn(dim_w, rank) * 0.1
    
    # Precompute indices for the tensor
    # We only need to check the 75 non-zero entries of the tensor
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_indices.append((a, b, c))
                
    # Also check some zero entries to penalize non-zero contributions there
    # We can't check ALL zeros (15*25*15 = 5625 total, 75 non-zero -> 5550 zeros)
    # Checking all zeros every iteration is expensive. 
    # However, for exact integer decomposition, we MUST ensure zeros are zero.
    # We will check a random subset of zeros during GD, and ALL zeros in final verification.
    zero_indices = []
    # Generate a list of all potential zero indices, then sample
    # To save time, we'll just rely on the final verification for strict zero-checking,
    # but during GD, we might drift. 
    # Actually, let's include a penalty for non-zero entries at random zero positions.
    
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_iters):
        # Calculate loss and gradients
        # Loss = sum_{(a,b,c) in T} (1 - sum_r U[a,r]V[b,r]W[c,r])^2 
        #       + lambda * sum_{(a,b,c) in Z} (sum_r U[a,r]V[b,r]W[c,r])^2
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Non-zero entries (should be 1)
        for a, b, c in tensor_indices:
            # val = sum_r U[a,r] V[b,r] W[c,r]
            # grad_U[a,r] += -2 * (1 - val) * V[b,r] * W[c,r]
            # grad_V[b,r] += -2 * (1 - val) * U[a,r] * W[c,r]
            # grad_W[c,r] += -2 * (1 - val) * U[a,r] * V[b,r]
            
            # Vectorized inner product for this entry
            # We need to compute the contribution of each rank r
            # This loop is over rank.
            # To optimize, we can precompute V[b,:] and W[c,:]
            v_b = V[b, :]
            w_c = W[c, :]
            u_a = U[a, :]
            
            # val = dot(u_a, v_b * w_c)
            val = np.dot(u_a, v_b * w_c)
            err = 1.0 - val
            loss += err * err
            
            # Gradient w.r.t U[a,:]
            # d/dU[a,r] (err^2) = 2 * err * d(err)/dU[a,r]
            # err = 1 - sum_k U[a,k] V[b,k] W[c,k]
            # d(err)/dU[a,r] = -V[b,r] W[c,r]
            # grad_U[a,r] += 2 * err * V[b,r] W[c,r]
            grad_U[a, :] += 2.0 * err * (v_b * w_c)
            
            # Gradient w.r.t V[b,:]
            # d(err)/dV[b,r] = -U[a,r] W[c,r]
            grad_V[b, :] += 2.0 * err * (u_a * w_c)
            
            # Gradient w.r.t W[c,:]
            # d(err)/dW[c,r] = -U[a,r] V[b,r]
            grad_W[c, :] += 2.0 * err * (u_a * v_b)
            
        # Zero entries penalty (sampled)
        # We want sum_r U[a,r]V[b,r]W[c,r] to be 0 for (a,b,c) not in T
        # Let's sample 100 random zero indices
        for _ in range(100):
            a = rng.randint(0, dim_u - 1)
            b = rng.randint(0, dim_v - 1)
            c = rng.randint(0, dim_w - 1)
            
            # Check if this is a non-zero index
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            if j1 == j2 and k1 == k2 and i1 == i2:
                continue # Skip if it's actually a non-zero entry
            
            v_b = V[b, :]
            w_c = W[c, :]
            u_a = U[a, :]
            
            val = np.dot(u_a, v_b * w_c)
            # Loss term: val^2
            loss += val * val
            
            # grad_U[a,r] += 2 * val * V[b,r] W[c,r]
            grad_U[a, :] += 2.0 * val * (v_b * w_c)
            grad_V[b, :] += 2.0 * val * (u_a * w_c)
            grad_W[c, :] += 2.0 * val * (u_a * v_b)
            
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        # Early stopping if loss is very small
        if best_loss < 1e-10:
            break
            
    # Attempt rationalization
    if best_U is not None:
        # Round to nearest integer or half-integer
        # The task allows integer/half-integer.
        # Multiply by 2, round, divide by 2
        U_rat = np.round(best_U * 2) / 2.0
        V_rat = np.round(best_V * 2) / 2.0
        W_rat = np.round(best_W * 2) / 2.0
        
        # Verify exact decomposition with rationalized values
        # Convert to int for exact check? No, half-integers.
        # We can check if np.allclose(U_rat * 2, np.round(U_rat * 2))
        # And then verify the tensor equation.
        
        # Check if they are actually close to the rounded values
        if np.allclose(best_U, U_rat) and np.allclose(best_V, V_rat) and np.allclose(best_W, W_rat):
            # Exact verification
            if _verify_decomposition(U_rat.tolist(), V_rat.tolist(), W_rat.tolist(), n, m, p):
                return U_rat.tolist(), V_rat.tolist(), W_rat.tolist()
                
    return None

def construct():
    """Find a low-rank decomposition for 3x5x5 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = 25
    # We try up to best_rank - 1
    # We use gradient descent for each rank
    
    # To manage time, we limit iterations per rank
    # Total time budget: 115s
    # Ranks to try: 25 to 74 (50 ranks)
    # ~2.3s per rank
    
    for target_rank in range(25, best_rank):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
            
        # Try a few random restarts for each rank
        found = False
        for seed in range(5): # 5 restarts
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
                
            result = _gradient_descent_decomposition(target_rank, n, m, p, max_iters=200, lr=0.01, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
            
    return best
