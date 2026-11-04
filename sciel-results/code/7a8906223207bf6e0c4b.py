# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Build target tensor
    def build_target_tensor(n, m, p):
        T = np.zeros((n*m, m*p, n*p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        return T
    
    T_target = build_target_tensor(n, m, p)
    
    # Schoolbook solution
    def schoolbook_solution(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook_solution(n, m, p)
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify_decomposition(U, V, W, T_target, tol=1e-6):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Check if entries are integers or half-integers
        # Scale by 2 to check integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        # Check if close to integers
        if not np.allclose(U2, np.round(U2)): return False, 1.0
        if not np.allclose(V2, np.round(V2)): return False, 1.0
        if not np.allclose(W2, np.round(W2)): return False, 1.0
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.abs(T_approx - T_target)
        max_diff = np.max(diff)
        return max_diff < tol, max_diff

    # Local Search for lower rank
    def local_search(rank, n, m, p, T_target, max_iters=5000, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize with small random integers/half-integers
        # Scale by 2 for integer arithmetic in search
        U = np.random.randint(-2, 3, size=(n*m, rank)) * 0.5
        V = np.random.randint(-2, 3, size=(m*p, rank)) * 0.5
        W = np.random.randint(-2, 3, size=(p*n, rank)) * 0.5
        
        def get_error(U, V, W):
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            return np.sum((T_approx - T_target)**2)
        
        current_err = get_error(U, V, W)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random matrix and a random entry
            mat_idx = np.random.randint(0, 3)
            if mat_idx == 0:
                r, c = np.random.randint(0, n*m), np.random.randint(0, rank)
                old_val = U[r, c]
            elif mat_idx == 1:
                r, c = np.random.randint(0, m*p), np.random.randint(0, rank)
                old_val = V[r, c]
            else:
                r, c = np.random.randint(0, p*n), np.random.randint(0, rank)
                old_val = W[r, c]
            
            # Try to change to a nearby half-integer
            # Neighbors: old_val +/- 0.5, old_val +/- 1.0
            neighbors = [old_val - 0.5, old_val + 0.5, old_val - 1.0, old_val + 1.0]
            
            best_neighbor = None
            best_neighbor_err = current_err
            
            for new_val in neighbors:
                # Apply change
                if mat_idx == 0:
                    U[r, c] = new_val
                elif mat_idx == 1:
                    V[r, c] = new_val
                else:
                    W[r, c] = new_val
                
                new_err = get_error(U, V, W)
                
                if new_err < best_neighbor_err:
                    best_neighbor_err = new_err
                    best_neighbor = new_val
                else:
                    # Revert
                    if mat_idx == 0:
                        U[r, c] = old_val
                    elif mat_idx == 1:
                        V[r, c] = old_val
                    else:
                        W[r, c] = old_val
            
            if best_neighbor is not None:
                # Apply the best change
                if mat_idx == 0:
                    U[r, c] = best_neighbor
                elif mat_idx == 1:
                    V[r, c] = best_neighbor
                else:
                    W[r, c] = best_neighbor
                current_err = best_neighbor_err
                
                if current_err < 1e-6:
                    valid, _ = verify_decomposition(U, V, W, T_target)
                    if valid:
                        return U, V, W
            else:
                # Stuck, restart or break? For now, break to save time
                # or try a random jump
                if np.random.random() < 0.01:
                    # Random restart for this trial
                    U = np.random.randint(-2, 3, size=(n*m, rank)) * 0.5
                    V = np.random.randint(-2, 3, size=(m*p, rank)) * 0.5
                    W = np.random.randint(-2, 3, size=(p*n, rank)) * 0.5
                    current_err = get_error(U, V, W)
        
        return None

    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n*p, m*p, n*m)
    
    # We know 60 works. Try to find < 60.
    # Start from 59 down to lower_bound? Or up?
    # Usually, it's easier to find a solution if we start close to the known solution.
    # But local search from random init is hard.
    # Let's try a few ranks.
    
    for rank in range(min(60, lower_bound + 10), lower_bound - 1, -1):
        if rank >= rank_sb:
            continue
        if time.time() > deadline:
            break
            
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
            res = local_search(rank, n, m, p, T_target, max_iters=1000, seed=trial*100+rank)
            if res is not None:
                U, V, W = res
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        if found:
            break

    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
