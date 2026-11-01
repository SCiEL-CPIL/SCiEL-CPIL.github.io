# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, tol=1e-9):
    """Check if the decomposition exactly decomposes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                uv = u_col[i * m + j] * v_col[j * p:]
                w_idx = np.array([k * n + i for k in range(p)])
                T[i, j, :] += uv * w_col[w_idx]
    return np.allclose(T, 1.0, atol=tol)

def _is_half_integer(arr, tol=1e-8):
    """Check if all entries are integer or half-integer."""
    arr = np.array(arr, dtype=float)
    doubled = 2.0 * arr
    nearest = np.round(doubled)
    return np.allclose(doubled, nearest, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = None
    best_rank = n * m * p
    
    def try_rank(rank, num_trials=50):
        nonlocal best_result, best_rank
        for trial in range(num_trials):
            if time.time() > deadline:
                return False
            np.random.seed(12345 + trial)
            
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U = np.random.choice(vals, size=(n * m, rank))
            V = np.random.choice(vals, size=(m * p, rank))
            
            W = np.zeros((p * n, rank))
            success = True
            for i in range(n):
                for k in range(p):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        M_pinv = np.linalg.pinv(M)
                        w = M_pinv @ np.ones(m)
                    except:
                        success = False
                        break
                    W[k * n + i, :] = w
                    if not _is_half_integer(w, tol=1e-7):
                        success = False
                        break
                if not success:
                    break
            
            if success:
                if _check_decomposition(U, V, W, tol=1e-8):
                    U = np.round(2 * U) / 2
                    V = np.round(2 * V) / 2
                    W = np.round(2 * W) / 2
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    return True
        
        return False
    
    # Try ranks from 49 down to 45 with fewer trials to stay within time limits
    for rank in range(49, 44, -1):
        if time.time() > deadline:
            break
        if try_rank(rank, num_trials=20):
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()

# EVOLVE-BLOCK-END
