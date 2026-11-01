import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Build the Strassen rank-28 solution as a fallback
    U, V, W = _build_strassen_28()
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = 28
    
    # Try to find a lower rank decomposition (target 20)
    # using a structural optimization approach
    result_20 = _search_rank_20(deadline)
    if result_20 is not None:
        return result_20
        
    # If rank 20 not found, try rank 21, 22, etc. with more time
    for target_rank in [21, 22, 23, 24]:
        if time.time() > deadline - 10:
            break
        result = _search_rank_target(target_rank, deadline)
        if result is not None:
            if target_rank < best_rank:
                best_rank = target_rank
                best_result = result
            break # Found a valid lower rank, return it
    
    return best_result

def _build_strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    
    # Compute T_hat[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (4, 16, rank)
    T_hat = np.einsum('abr,cr->abc', UV, W)  # (4, 16, 16)
    
    # Check non-zero entries: T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i)
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                if abs(T_hat[a, b, c] - 1) > tol:
                    return False
    
    # Check zero entries
    for a in range(num_a):
        for b in range(num_b):
            for c in range(num_c):
                i = a // 2
                j = a % 2
                j2 = b // 8
                k = b % 8
                k2 = c // 2
                i2 = c % 2
                if not (i == i2 and j == j2 and k == k2):
                    if abs(T_hat[a, b, c]) > tol:
                        return False
    
    return True

def _search_rank_20(deadline):
    """
    Structural search for Rank 20.
    Uses random initialization of U and V, then solves for W via least squares,
    rounds to half-integers, and verifies.
    """
    start = time.time()
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    target_rank = 20
    
    # Precompute the target tensor T
    T = np.zeros((num_a, num_b, num_c))
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() - start > deadline - 5:
            break
            
        rng = np.random.default_rng(seed=restart)
        
        # Initialize U and V with random half-integers
        # Use a smaller range to increase chance of small integers
        U = rng.integers(-2, 3, size=(num_a, target_rank)) * 0.5
        V = rng.integers(-2, 3, size=(num_b, target_rank)) * 0.5
        
        # Construct the matrix A such that W = A^{-1} b
        # T_flat = sum_r U[:,r] * V[:,r] * W[:,r]
        # This is linear in W.
        # Let M be a matrix where M[c, r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r]
        # Then sum_r M[c, r] W[r] = T_flat? No.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For each c, T[:, :, c] = sum_r (U[:,r] * V[:,r]) W[c,r]
        # Let X_c be the vectorization of T[:, :, c].
        # Let Y_r be the vectorization of U[:,r] * V[:,r].
        # X_c = sum_r Y_r W[c,r]
        # So X = Y W^T where X is (num_c, num_a*num_b) and Y is (num_a*num_b, target_rank)
        
        X = T.reshape(num_c, -1).T # (1024, 16)
        # Y_r = vec(U[:,r] * V[:,r])
        Y = np.zeros((num_a * num_b, target_rank))
        for r in range(target_rank):
            Y[:, r] = (U[:, r] * V[:, r]).flatten()
            
        # Solve Y W^T = X for W^T
        # W^T = Y^+ X
        try:
            W_solution, residuals, rank_y, sv = np.linalg.lstsq(Y, X, rcond=None)
            W_candidate = W_solution.T # (16, 20)
        except np.linalg.LinAlgError:
            continue
            
        # Round to nearest half-integer
        W_rounded = np.round(W_candidate * 2) / 2
        
        # Verify
        if _verify(U, V, W_rounded, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W_rounded.tolist()}
            
    return None

def _search_rank_target(target_rank, deadline):
    """
    Generalized search for a specific target rank.
    """
    start = time.time()
    n, m, p = 2, 2, 8
    num_a, num_b, num_c = 4, 16, 16
    
    T = np.zeros((num_a, num_b, num_c))
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                T[a, b, c] = 1.0
                
    X = T.reshape(num_c, -1).T # (1024, 16)
    
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() - start > deadline - 5:
            break
            
        rng = np.random.default_rng(seed=restart + 1000)
        
        U = rng.integers(-2, 3, size=(num_a, target_rank)) * 0.5
        V = rng.integers(-2, 3, size=(num_b, target_rank)) * 0.5
        
        Y = np.zeros((num_a * num_b, target_rank))
        for r in range(target_rank):
            Y[:, r] = (U[:, r] * V[:, r]).flatten()
            
        try:
            W_solution, residuals, rank_y, sv = np.linalg.lstsq(Y, X, rcond=None)
            W_candidate = W_solution.T
        except np.linalg.LinAlgError:
            continue
            
        W_rounded = np.round(W_candidate * 2) / 2
        
        if _verify(U, V, W_rounded, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W_rounded.tolist()}
            
    return None
