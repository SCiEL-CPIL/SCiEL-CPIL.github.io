import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # 1. Construct Schoolbook Decomposition (Safe Fallback)
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # 2. Try to find a lower rank decomposition
    # We target rank 29 (one less than schoolbook)
    # If we find it, score = 30/29 > 1.0
    # If not, return schoolbook.
    
    target_rank = rank_schoolbook - 1
    if target_rank < 1:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    # Initialize random matrices for the lower rank attempt
    # Use a seeded RNG for reproducibility in debugging, but random is fine
    rng = np.random.default_rng(42)
    
    U = rng.normal(size=(n * m, target_rank))
    V = rng.normal(size=(m * p, target_rank))
    W = rng.normal(size=(p * n, target_rank))
    
    # Define the tensor T (sparse)
    # T[a, b, c] = 1 if indices are consistent, else 0
    # We only need to check the non-zero entries and a sample of zero entries
    # Actually, we need to satisfy ALL entries.
    # Non-zero entries: 30
    # Zero entries: 900 - 30 = 870
    
    # List of non-zero indices (a, b, c)
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zero_indices.append((a, b, c))
                
    # List of zero indices (sample if too many, but 870 is small enough to check all)
    all_indices = [(a, b, c) for a in range(n * m) for b in range(m * p) for c in range(p * n)]
    non_zero_set = set(non_zero_indices)
    zero_indices = [idx for idx in all_indices if idx not in non_zero_set]
    
    # Optimization loop
    # Simple Gradient Descent
    # Error = sum (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
    
    lr = 0.01
    max_iter = 5000
    start_time = time.time()
    deadline = start_time + 100 # Leave time for verification and return
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Accumulate errors and gradients for non-zero and zero entries
        # We can vectorize this partially, but explicit loop is safer for clarity and small size
        
        # To speed up, we can sample a subset of zero entries if the full set is slow
        # But 870 * 29 is small.
        
        error = 0.0
        
        # Process non-zero entries
        for a, b, c in non_zero_indices:
            # Compute sum
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            err = s - 1.0
            error += err * err
            
            # Gradients
            # d/dU[a,r] = err * V[b,r] * W[c,r]
            dU[a, :] += err * V[b, :] * W[c, :]
            dV[b, :] += err * U[a, :] * W[c, :]
            dW[c, :] += err * U[a, :] * V[b, :]
            
        # Process zero entries (sample 100 if too slow, but let's try all)
        for a, b, c in zero_indices:
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            err = s - 0.0
            error += err * err
            
            dU[a, :] += err * V[b, :] * W[c, :]
            dV[b, :] += err * U[a, :] * W[c, :]
            dW[c, :] += err * U[a, :] * V[b, :]
            
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if error < 1e-6:
            # Check if we can round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded solution
            if _verify(U_rounded, V_rounded, W_rounded, non_zero_indices, zero_indices, n, m, p):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
            # If not rounded, check if raw is valid enough?
            # The problem requires exact decomposition.
            # So we must find an exact one.
            
    # If we found a low error but not exact, we might still try to return it?
    # No, "exactly decompose".
    # So if we didn't find an exact one, return schoolbook.
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def _verify(U, V, W, non_zero_indices, zero_indices, n, m, p):
    # Verify non-zero
    for a, b, c in non_zero_indices:
        if abs(np.sum(U[a, :] * V[b, :] * W[c, :]) - 1.0) > 1e-9:
            return False
    # Verify zero
    for a, b, c in zero_indices:
        if abs(np.sum(U[a, :] * V[b, :] * W[c, :])) > 1e-9:
            return False
    return True
