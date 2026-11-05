import time
import numpy as np

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check T_{i*m+j, j*p+k, k*n+i} = 1
    # And all other entries 0? No, just the specific ones must be 1.
    # The tensor is sparse.
    # We just need to ensure the sum reconstructs the 1s.
    # The standard check is to compute the full tensor and compare.
    
    # Efficient check:
    # For each (i, j, k), sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1.
    # For other (a, b, c), it should be 0.
    # But checking all (a, b, c) is expensive (10*25*10 = 2500 checks).
    # We can just check the 50 non-zero entries.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _find_rank_r_decomp_1xn(n, r, max_iters=1000, lr=0.01):
    """
    Attempt to find a rank-r decomposition for 1 x n x n multiplication.
    T_{j, j'*n+k, k} = 1 if j=j'.
    """
    # Target tensor T is sparse.
    # We minimize || Sum_r u_r v_r^T w_r^T - T ||^2
    
    # Initialize factors
    # U: (n, r), V: (n*n, r), W: (n, r)
    U = np.random.randn(n, r) * 0.1
    V = np.random.randn(n * n, r) * 0.1
    W = np.random.randn(n, r) * 0.1
    
    # Compute gradient
    # Loss = Sum_{a,b,c} (Sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c])^2
    # dL/dU[a,r] = 2 * Sum_{b,c} (Err[a,b,c] * V[b,r] * W[c,r])
    # where Err[a,b,c] = (Sum_{r'} U[a,r'] V[b,r'] W[c,r'] - T[a,b,c])
    
    # To speed up, we can use the fact that T is sparse.
    # T[j, j*n+k, k] = 1.
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
            
        # Compute current tensor (sparse check)
        # We only need the gradient at the non-zero points and maybe some random points?
        # No, gradient descent needs the full gradient or a valid stochastic gradient.
        # Given small size (5x25x5), we can compute the full tensor.
        
        # Tensor shape (n, n*n, n)
        T_current = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target T
        T_target = np.zeros((n, n * n, n))
        for j in range(n):
            for k in range(n):
                T_target[j, j * n + k, k] = 1.0
        
        Err = T_current - T_target
        Loss = np.sum(Err ** 2)
        
        if Loss < best_loss:
            best_loss = Loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if Loss < 1e-8:
            break
            
        # Compute Gradients
        # dU[a, r] = 2 * Sum_{b,c} Err[a,b,c] * V[b,r] * W[c,r]
        # This can be done with einsum
        # G_U = 2 * einsum('abc,br,cr->ar', Err, V, W)
        # G_V = 2 * einsum('abc,ar,cr->br', Err, U, W)
        # G_W = 2 * einsum('abc,ar,br->cr', Err, U, V)
        
        G_U = 2 * np.einsum('abc,br,cr->ar', Err, V, W)
        G_V = 2 * np.einsum('abc,ar,cr->br', Err, U, W)
        G_W = 2 * np.einsum('abc,ar,br->cr', Err, U, V)
        
        # Update
        U -= lr * G_U
        V -= lr * G_V
        W -= lr * G_W
        
        # Optional: Renormalize to prevent scale drift
        # U /= np.linalg.norm(U, axis=0, keepdims=True)
        # V /= np.linalg.norm(V, axis=0, keepdims=True)
        # W /= np.linalg.norm(W, axis=0, keepdims=True)
        
    return best_U, best_V, best_W, best_loss

def _try_rank_r(n, m, p, r):
    if r >= n * m * p:
        return None
    
    # Strategy: Decompose into n independent 1 x m x p tensors.
    # For 2x5x5, we need rank r/2 for each 1x5x5.
    # r must be even.
    if r % 2 != 0:
        return None
        
    r_half = r // 2
    
    # Try to find rank r_half decomposition for 1x5x5
    # We try a few times with different seeds
    for attempt in range(5):
        np.random.seed(attempt * 1000)
        U1, V1, W1, loss = _find_rank_r_decomp_1xn(m, r_half, max_iters=500, lr=0.005)
        
        if loss < 1e-6:
            # Construct full decomposition
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            for i in range(n):
                # Block i
                # U rows i*m to (i+1)*m
                # W rows i*p to (i+1)*p ? No, W index is k*n+i.
                # W row index: k*n + i.
                # For fixed i, the rows are i, n+i, 2n+i, ...
                # Wait, W shape is (p*n, rank).
                # Index is k*n + i.
                # So for i=0, rows are 0, 2, 4, 6, 8.
                # For i=1, rows are 1, 3, 5, 7, 9.
                
                # Copy U1 to U block
                U[i * m : (i + 1) * m, i * r_half : (i + 1) * r_half] = U1
                
                # Copy V1 to V block
                V[:, i * r_half : (i + 1) * r_half] = V1
                
                # Copy W1 to W block
                # W rows for this i are at indices k*n + i for k in 0..p-1
                # We need to place W1 (shape n x r_half) into these rows.
                # W1 row j corresponds to C index j?
                # In 1xn x n, C index is k (0..n-1).
                # In full tensor, C index is k*n + i.
                # So W1[j, :] should go to W[j*n + i, :].
                
                for k in range(p):
                    W[k * n + i, i * r_half : (i + 1) * r_half] = W1[k, :]
            
            # Verify
            if _verify(n, m, p, U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try rank 46 (2 * 23)
    result = _try_rank_r(n, m, p, 46)
    if result is not None:
        return result
        
    # Try rank 48 (2 * 24)
    result = _try_rank_r(n, m, p, 48)
    if result is not None:
        return result
        
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
