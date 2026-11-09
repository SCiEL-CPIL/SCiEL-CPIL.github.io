# EVOLVE-BLOCK-START
def _check_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Convention: W[k*n + i, c]
    The tensor T[i,j,k] = 1 should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    """
    r = U.shape[1]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    if not np.allclose(T, 1.0, atol=1e-9, rtol=1e-9):
        return False
    return True

def _make_schoolbook(n, m, p):
    """Create the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_local_search(n, m, p, r, max_iter=100, seed=0, deadline=None):
    """
    Try to find a rank-r decomposition using local search on half-integer values.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random half-integers
    U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
    V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, r))
    W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, r))
    
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    def compute_error(U_r, V_r, W_r):
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.sum((T - 1.0)**2)
    
    current_error = compute_error(U_reshaped, V_reshaped, W_reshaped)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        tensor_choice = rng.randint(3)
        
        if tensor_choice == 0:
            idx = rng.randint(n * m * r)
            i, j, c = np.unravel_index(idx, (n, m, r))
            old_val = U_reshaped[i, j, c]
            for new_val in [-1, -0.5, 0, 0.5, 1]:
                if new_val == old_val:
                    continue
                U_reshaped[i, j, c] = new_val
                new_error = compute_error(U_reshaped, V_reshaped, W_reshaped)
                if new_error < current_error:
                    current_error = new_error
                    break
                else:
                    U_reshaped[i, j, c] = old_val
        elif tensor_choice == 1:
            idx = rng.randint(m * p * r)
            j, k, c = np.unravel_index(idx, (m, p, r))
            old_val = V_reshaped[j, k, c]
            for new_val in [-1, -0.5, 0, 0.5, 1]:
                if new_val == old_val:
                    continue
                V_reshaped[j, k, c] = new_val
                new_error = compute_error(U_reshaped, V_reshaped, W_reshaped)
                if new_error < current_error:
                    current_error = new_error
                    break
                else:
                    V_reshaped[j, k, c] = old_val
        else:
            idx = rng.randint(p * n * r)
            k, i, c = np.unravel_index(idx, (p, n, r))
            old_val = W_reshaped[k, i, c]
            for new_val in [-1, -0.5, 0, 0.5, 1]:
                if new_val == old_val:
                    continue
                W_reshaped[k, i, c] = new_val
                new_error = compute_error(U_reshaped, V_reshaped, W_reshaped)
                if new_error < current_error:
                    current_error = new_error
                    break
                else:
                    W_reshaped[k, i, c] = old_val
        
        if current_error < 1e-9:
            U = U_reshaped.reshape(n * m, r)
            V = V_reshaped.reshape(m * p, r)
            W = W_reshaped.reshape(p * n, r)
            if _check_decomposition(U, V, W, n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
            
    U = U_reshaped.reshape(n * m, r)
    V = V_reshaped.reshape(m * p, r)
    W = W_reshaped.reshape(p * n, r)
    if _check_decomposition(U, V, W, n, m, p):
        return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U, V, W = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try ranks 104, 103, 102
    for r in [104, 103, 102]:
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            break
            
        found = False
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            result = _try_local_search(n, m, p, r, max_iter=50, seed=seed, deadline=deadline-20)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            continue
            
    return best_result
# EVOLVE-BLOCK-END
