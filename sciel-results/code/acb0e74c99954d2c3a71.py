<<<<<<< SEARCH
def _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline):
    """Try to reduce the rank starting from a baseline decomposition."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    # Try reducing rank by dropping columns and re-optimizing
    for target_rank in range(rank - 1, max(45, rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        for init_seed in range(4):
            if time.time() > deadline:
                break
            
            if init_seed == 0:
                U_init = Ua[:, :target_rank]
                V_init = Va[:, :target_rank]
                W_init = Wa[:, :target_rank]
            elif init_seed == 1:
                U_init = Ua[:, rank-target_rank:]
                V_init = Va[:, rank-target_rank:]
                W_init = Wa[:, rank-target_rank:]
            elif init_seed == 2:
                np.random.seed(init_seed * 100 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx]
                V_init = Va[:, idx]
                W_init = Wa[:, idx]
            else:
                # Random linear combination
                np.random.seed(init_seed * 200 + target_rank)
                U_init = Ua @ np.random.randn(rank, target_rank) * 0.5
                V_init = Va @ np.random.randn(rank, target_rank) * 0.5
                W_init = Wa @ np.random.randn(rank, target_rank) * 0.5
            
            result = _als_decompose(n, m, p, target_rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=200, polish_iters=100, deadline=deadline)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
    
    return best_result
=======
def _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline):
    """Try to reduce the rank starting from a baseline decomposition."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    # Try reducing rank by dropping columns and re-optimizing
    for target_rank in range(rank - 1, max(45, rank - 25), -1):
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        for init_seed in range(6):
            if time.time() > deadline:
                break
            
            if init_seed == 0:
                U_init = Ua[:, :target_rank]
                V_init = Va[:, :target_rank]
                W_init = Wa[:, :target_rank]
            elif init_seed == 1:
                U_init = Ua[:, rank-target_rank:]
                V_init = Va[:, rank-target_rank:]
                W_init = Wa[:, rank-target_rank:]
            elif init_seed == 2:
                np.random.seed(init_seed * 100 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx]
                V_init = Va[:, idx]
                W_init = Wa[:, idx]
            elif init_seed == 3:
                # Random linear combination
                np.random.seed(init_seed * 200 + target_rank)
                U_init = Ua @ np.random.randn(rank, target_rank) * 0.5
                V_init = Va @ np.random.randn(rank, target_rank) * 0.5
                W_init = Wa @ np.random.randn(rank, target_rank) * 0.5
            elif init_seed == 4:
                # Sparse initialization: keep columns with largest norms
                col_norms = np.linalg.norm(Ua, axis=0)
                idx = np.argsort(col_norms)[::-1][:target_rank]
                U_init = Ua[:, idx]
                V_init = Va[:, idx]
                W_init = Wa[:, idx]
            else:
                # Structured: alternate between first and last half
                half = target_rank // 2
                idx = list(range(half)) + list(range(rank - (target_rank - half), rank))
                U_init = Ua[:, idx]
                V_init = Va[:, idx]
                W_init = Wa[:, idx]
            
            result = _als_decompose(n, m, p, target_rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=150, polish_iters=80, deadline=deadline)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
    
    return best_result
>>>>>>> REPLACE

