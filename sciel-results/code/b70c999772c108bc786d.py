import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    """Build the matrix multiplication tensor for 3x3 * 3x4."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor."""
    T = _build_tensor()
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _is_valid_integer_solution(U, V, W, tol=1e-9):
    """Check if U, V, W have integer or half-integer entries and verify decomposition."""
    for mat in [U, V, W]:
        doubled = mat * 2
        rounded = np.round(doubled)
        if not np.allclose(doubled, rounded, atol=tol):
            return False
    return _verify_decomposition(U, V, W, tol)

def _get_rank_34_baseline():
    """Return the known rank-34 decomposition as numpy arrays."""
    n, m, p = DIMENSIONS
    rank = 34
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Strassen Left (for columns 0,1 of B)
    U[0, col]=1; U[4, col]=1; V[0, col]=1; V[5, col]=1; W[0, col]=1; W[4, col]=1; col+=1
    U[3, col]=1; U[4, col]=1; V[0, col]=1; W[1, col]=1; W[4, col]=-1; col+=1
    U[0, col]=1; V[1, col]=1; V[5, col]=-1; W[3, col]=1; W[4, col]=1; col+=1
    U[4, col]=1; V[4, col]=1; V[0, col]=-1; W[0, col]=1; W[1, col]=1; col+=1
    U[0, col]=1; U[1, col]=1; V[5, col]=1; W[0, col]=-1; W[3, col]=1; col+=1
    U[3, col]=1; U[0, col]=-1; V[0, col]=1; V[1, col]=1; W[4, col]=1; col+=1
    U[1, col]=1; U[4, col]=-1; V[4, col]=1; V[5, col]=1; W[0, col]=1; col+=1
    # A12*B2L
    U[2, col]=1; V[8, col]=1; W[0, col]=1; col+=1
    U[2, col]=1; V[9, col]=1; W[3, col]=1; col+=1
    U[5, col]=1; V[8, col]=1; W[1, col]=1; col+=1
    U[5, col]=1; V[9, col]=1; W[4, col]=1; col+=1
    # A21*B1L
    U[6, col]=1; V[0, col]=1; W[2, col]=1; col+=1
    U[6, col]=1; V[1, col]=1; W[5, col]=1; col+=1
    U[7, col]=1; V[4, col]=1; W[2, col]=1; col+=1
    U[7, col]=1; V[5, col]=1; W[5, col]=1; col+=1
    # A22*B2L
    U[8, col]=1; V[8, col]=1; W[2, col]=1; col+=1
    U[8, col]=1; V[9, col]=1; W[5, col]=1; col+=1
    # Strassen Right (for columns 2,3 of B)
    U[0, col]=1; U[4, col]=1; V[2, col]=1; V[7, col]=1; W[6, col]=1; W[10, col]=1; col+=1
    U[3, col]=1; U[4, col]=1; V[2, col]=1; W[7, col]=1; W[10, col]=-1; col+=1
    U[0, col]=1; V[3, col]=1; V[7, col]=-1; W[9, col]=1; W[10, col]=1; col+=1
    U[4, col]=1; V[6, col]=1; V[2, col]=-1; W[6, col]=1; W[7, col]=1; col+=1
    U[0, col]=1; U[1, col]=1; V[7, col]=1; W[6, col]=-1; W[9, col]=1; col+=1
    U[3, col]=1; U[0, col]=-1; V[2, col]=1; V[3, col]=1; W[10, col]=1; col+=1
    U[1, col]=1; U[4, col]=-1; V[6, col]=1; V[7, col]=1; W[6, col]=1; col+=1
    # A12*B2R
    U[2, col]=1; V[10, col]=1; W[6, col]=1; col+=1
    U[2, col]=1; V[11, col]=1; W[9, col]=1; col+=1
    U[5, col]=1; V[10, col]=1; W[7, col]=1; col+=1
    U[5, col]=1; V[11, col]=1; W[10, col]=1; col+=1
    # A21*B1R
    U[6, col]=1; V[2, col]=1; W[8, col]=1; col+=1
    U[6, col]=1; V[3, col]=1; W[11, col]=1; col+=1
    U[7, col]=1; V[6, col]=1; W[8, col]=1; col+=1
    U[7, col]=1; V[7, col]=1; W[11, col]=1; col+=1
    # A22*B2R
    U[8, col]=1; V[10, col]=1; W[8, col]=1; col+=1
    U[8, col]=1; V[11, col]=1; W[11, col]=1; col+=1
    
    return U, V, W

def _try_remove_column(U, V, W, col_idx):
    """Try removing a single column and see if the remaining still decomposes the tensor."""
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    mask = np.ones(rank, dtype=bool)
    mask[col_idx] = False
    U_sub = U[:, mask]
    V_sub = V[:, mask]
    W_sub = W[:, mask]
    
    if _verify_decomposition(U_sub, V_sub, W_sub):
        return (U_sub, V_sub, W_sub)
    return None

def _greedy_column_removal(U, V, W, max_time=5.0):
    """Greedy: repeatedly try to remove one column at a time."""
    start_time = time.time()
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for col in range(current_U.shape[1]):
            if time.time() - start_time > max_time:
                break
            result = _try_remove_column(current_U, current_V, current_W, col)
            if result is not None:
                current_U, current_V, current_W = result
                improved = True
                break
    
    return current_U, current_V, current_W

def _random_subset_test(U, V, W, target_rank, n_trials=100, max_time=5.0, seed=42):
    """Test random subsets of columns of size target_rank."""
    start_time = time.time()
    current_rank = U.shape[1]
    np.random.seed(seed)
    
    for trial in range(n_trials):
        if time.time() - start_time > max_time:
            break
        
        cols = np.random.choice(current_rank, size=target_rank, replace=False)
        U_sub = U[:, cols]
        V_sub = V[:, cols]
        W_sub = W[:, cols]
        
        if _verify_decomposition(U_sub, V_sub, W_sub):
            return (U_sub, V_sub, W_sub)
    
    return None

def _structured_perturbation_search(U_base, V_base, W_base, target_rank, max_time=10.0, seed=42):
    """
    Try structured perturbations of the baseline and then reduce rank.
    Focus on half-integer perturbations that might enable column merging.
    """
    start_time = time.time()
    np.random.seed(seed)
    current_rank = U_base.shape[1]
    
    if target_rank >= current_rank:
        return None
    
    for attempt in range(200):
        if time.time() - start_time > max_time:
            break
        
        U = U_base.copy()
        V = V_base.copy()
        W = W_base.copy()
        
        # Apply small structured perturbations
        n_perturb = np.random.randint(1, 6)
        for _ in range(n_perturb):
            # Pick random entries to perturb
            u_row = np.random.randint(0, U.shape[0])
            v_row = np.random.randint(0, V.shape[0])
            w_row = np.random.randint(0, W.shape[0])
            u_col = np.random.randint(0, U.shape[1])
            v_col = np.random.randint(0, V.shape[1])
            w_col = np.random.randint(0, W.shape[1])
            
            # Half-integer perturbations
            U[u_row, u_col] += np.random.choice([-0.5, 0.5])
            V[v_row, v_col] += np.random.choice([-0.5, 0.5])
            W[w_row, w_col] += np.random.choice([-0.5, 0.5])
        
        # Try to reduce rank from perturbed version
        # First check if it's still valid
        if not _verify_decomposition(U, V, W):
            continue
        
        # Try greedy removal
        U_g, V_g, W_g = _greedy_column_removal(U, V, W, max_time=min(2.0, max_time - (time.time() - start_time)))
        
        if U_g.shape[1] <= target_rank:
            # Try to round to half-integers
            U_r = np.round(U_g * 2) / 2
            V_r = np.round(V_g * 2) / 2
            W_r = np.round(W_g * 2) / 2
            
            if _verify_decomposition(U_r, V_r, W_r) and _is_valid_integer_solution(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        
        # Also try random subset of target_rank
        if U_g.shape[1] > target_rank:
            result = _random_subset_test(U_g, V_g, W_g, target_rank, 
                                          n_trials=20, 
                                          max_time=min(1.0, max_time - (time.time() - start_time)),
                                          seed=seed + attempt)
            if result is not None:
                U_r, V_r, W_r = result
                U_r = np.round(U_r * 2) / 2
                V_r = np.round(V_r * 2) / 2
                W_r = np.round(W_r * 2) / 2
                if _verify_decomposition(U_r, V_r, W_r) and _is_valid_integer_solution(U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
    
    return None

def _try_merge_two_columns(U, V, W, col1, col2, max_time=2.0):
    """
    Try to merge two columns by finding a linear combination.
    If column j can be expressed as a combination of others, remove it.
    This is a more sophisticated approach: check if the tensor contribution
    of one column is in the span of the others.
    """
    start_time = time.time()
    rank = U.shape[1]
    
    # Build the tensor contribution of each column
    T = _build_tensor()
    T_flat = T.flatten()
    
    # Build matrix where each column is the flattened tensor contribution
    col_tensors = np.zeros((T_flat.size, rank))
    for r in range(rank):
        col_tensors[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
    
    # Check if column col2 is in the span of all columns except col1 and col2
    # Actually, check if we can eliminate col2 by expressing it in terms of others
    mask = np.ones(rank, dtype=bool)
    mask[col2] = False
    others = col_tensors[:, mask]
    
    # Try to solve: col_tensors[:, col2] ≈ others @ x
    try:
        x, residuals, rank_sv, sv = np.linalg.lstsq(others, col_tensors[:, col2], rcond=None)
        if residuals.size > 0 and residuals[0] < 1e-8:
            # col2 is in the span of others, so we can remove it
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            if _verify_decomposition(U_new, V_new, W_new):
                return (U_new, V_new, W_new)
    except:
        pass
    
    return None

def _linear_dependency_search(U, V, W, target_rank, max_time=10.0, seed=42):
    """
    Search for linear dependencies among columns to reduce rank.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(50):
        if time.time() - start_time > max_time:
            break
        if current_U.shape[1] <= target_rank:
            break
        
        # Try removing each column
        found = False
        for col in range(current_U.shape[1]):
            if time.time() - start_time > max_time:
                break
            
            result = _try_merge_two_columns(current_U, current_V, current_W, 
                                            (col+1) % current_U.shape[1], col,
                                            max_time=min(1.0, max_time - (time.time() - start_time)))
            if result is not None:
                current_U, current_V, current_W = result
                found = True
                break
        
        if not found:
            # Try random subset
            result = _random_subset_test(current_U, current_V, current_W, 
                                          min(target_rank, current_U.shape[1]-1),
                                          n_trials=30,
                                          max_time=min(2.0, max_time - (time.time() - start_time)),
                                          seed=seed + iteration)
            if result is not None:
                current_U, current_V, current_W = result
            else:
                break
    
    if current_U.shape[1] <= target_rank:
        U_r = np.round(current_U * 2) / 2
        V_r = np.round(current_V * 2) / 2
        W_r = np.round(current_W * 2) / 2
        if _verify_decomposition(U_r, V_r, W_r) and _is_valid_integer_solution(U_r, V_r, W_r):
            return (U_r, V_r, W_r)
    
    return None

def construct():
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Start with the known rank-34 solution
    U_base, V_base, W_base = _get_rank_34_baseline()
    best_result = {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    best_rank = 34
    
    # First, try greedy column removal on the baseline
    if time.time() < deadline - 30:
        U_g, V_g, W_g = _greedy_column_removal(U_base, V_base, W_base, max_time=5.0)
        if U_g.shape[1] < best_rank:
            U_r = np.round(U_g * 2) / 2
            V_r = np.round(V_g * 2) / 2
            W_r = np.round(W_g * 2) / 2
            if _verify_decomposition(U_r, V_r, W_r) and _is_valid_integer_solution(U_r, V_r, W_r):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = U_g.shape[1]
    
    # Try to find better decompositions using various search strategies
    for rank in range(best_rank - 1, 25, -1):
        if time.time() > deadline - 20:
            break
        
        remaining_time = deadline - time.time() - 15
        if remaining_time < 3.0:
            break
        
        # Strategy 1: Linear dependency search
        max_time1 = min(8.0, remaining_time * 0.4)
        result1 = _linear_dependency_search(U_base, V_base, W_base, rank, 
                                             max_time=max_time1, seed=42 + rank)
        if result1 is not None:
            best_result = {"U": result1[0].tolist(), "V": result1[1].tolist(), "W": result1[2].tolist()}
            best_rank = result1[0].shape[1]
            continue
        
        # Strategy 2: Structured perturbation search
        remaining_time = deadline - time.time() - 15
        if remaining_time < 3.0:
            break
        max_time2 = min(8.0, remaining_time * 0.4)
        result2 = _structured_perturbation_search(U_base, V_base, W_base, rank,
                                                    max_time=max_time2, seed=100 + rank)
        if result2 is not None:
            best_result = {"U": result2[0].tolist(), "V": result2[1].tolist(), "W": result2[2].tolist()}
            best_rank = result2[0].shape[1]
            continue
        
        # Strategy 3: Random subset test on baseline
        remaining_time = deadline - time.time() - 15
        if remaining_time < 2.0:
            break
        max_time3 = min(5.0, remaining_time * 0.3)
        result3 = _random_subset_test(U_base, V_base, W_base, rank,
                                       n_trials=50,
                                       max_time=max_time3,
                                       seed=200 + rank)
        if result3 is not None:
            U_r, V_r, W_r = result3
            U_r = np.round(U_r * 2) / 2
            V_r = np.round(V_r * 2) / 2
            W_r = np.round(W_r * 2) / 2
            if _verify_decomposition(U_r, V_r, W_r) and _is_valid_integer_solution(U_r, V_r, W_r):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = U_r.shape[1]
    
    return best_result
