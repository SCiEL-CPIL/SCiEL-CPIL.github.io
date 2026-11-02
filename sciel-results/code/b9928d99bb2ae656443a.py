# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _compute_error_fast(U, V, W, T):
    """Compute error efficiently using einsum."""
    T_est = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return float(np.sum((T_est - T) ** 2))


def construct():
    import time
    import random
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p  # 90
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = rank_schoolbook
    
    # Strategy: Try to reduce rank by merging pairs of schoolbook terms
    # Two terms (i,j,k) and (i',j',k') can be merged into a single rank-1 term
    # if the 4D tensor formed by their outer products has rank 1, which requires
    # specific index relationships.
    
    # More practically: try random rank-(r) initializations with structured patterns
    # and use accelerated local search.
    
    random.seed(123)
    np.random.seed(123)
    
    # Precompute for faster error evaluation
    # Use a sampling-based error for speed during search
    
    def compute_error_sampled(U, V, W, num_samples=500):
        """Compute error on a random sample of tensor entries for speed."""
        # Sample consistent and inconsistent entries
        indices = []
        # Consistent entries
        for _ in range(num_samples // 2):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            indices.append((i * m + j, j * p + k, k * n + i, 1.0))
        # Inconsistent entries
        for _ in range(num_samples // 2):
            a = random.randint(0, nm - 1)
            b = random.randint(0, mp - 1)
            c = random.randint(0, pn - 1)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            expected = 1.0 if (j2 == j and k2 == k and i2 == i) else 0.0
            indices.append((a, b, c, expected))
        
        err = 0.0
        for a, b, c, exp in indices:
            actual = float(np.dot(U[a, :], np.dot(V[b, :], W[c, :])))
            err += (actual - exp) ** 2
        return err
    
    def full_verify(U, V, W):
        """Full verification of the decomposition."""
        T_est = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return float(np.sum((T_est - T) ** 2)) < 1e-9
    
    # Try structured random search with various ranks
    for target_rank in range(rank_schoolbook - 1, 60, -1):
        if time.time() > deadline - 20.0:
            break
        
        max_trials = 50 if target_rank > 80 else 20
        
        for trial in range(max_trials):
            if time.time() > deadline - 20.0:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.3
            V = np.random.randn(mp, target_rank) * 0.3
            W = np.random.randn(pn, target_rank) * 0.3
            
            # Local search with coordinate descent
            err = compute_error_sampled(U, V, W)
            
            for step in range(200):
                if time.time() > deadline - 20.0:
                    break
                if err < 1e-6:
                    break
                
                # Pick random matrix and coordinate
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    arr = U
                    rows, cols = nm, target_rank
                elif mat_idx == 1:
                    arr = V
                    rows, cols = mp, target_rank
                else:
                    arr = W
                    rows, cols = pn, target_rank
                
                r_idx = random.randint(0, rows - 1)
                c_idx = random.randint(0, cols - 1)
                
                old_val = arr[r_idx, c_idx]
                best_pert = old_val
                best_err = err
                
                for delta in [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]:
                    arr[r_idx, c_idx] = old_val + delta
                    new_err = compute_error_sampled(U, V, W)
                    if new_err < best_err:
                        best_err = new_err
                        best_pert = old_val + delta
                
                arr[r_idx, c_idx] = best_pert
                err = best_err
            
            # Check if we found a valid solution
            if err < 1e-4:
                # Full verification
                if full_verify(U, V, W):
                    # Check for half-integer values
                    all_half = True
                    for arr in [U, V, W]:
                        if not np.allclose(arr * 2, np.round(arr * 2), atol=1e-6):
                            all_half = False
                            break
                    
                    if all_half:
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if full_verify(U_r, V_r, W_r):
                            if target_rank < best_rank:
                                best_U = U_r.copy()
                                best_V = V_r.copy()
                                best_W = W_r.copy()
                                best_rank = target_rank
                                break
                    else:
                        # Accept as-is if exact
                        if target_rank < best_rank:
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                            best_rank = target_rank
                            break
    
    # Return best solution
    if best_rank >= rank_schoolbook:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
