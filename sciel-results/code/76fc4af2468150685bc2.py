<<<<<<< SEARCH
def construct():
    deadline = time.time() + 110
    
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)
    
    # Try ranks from 45 to 51
    best_result = None
    
    for target_rank in range(45, 52):
        if time.time() > deadline - 10:
            break
        
        U, V, W = _als_decompose(T, target_rank, n_iter=200)
        
        if _verify_decomposition(U, V, W, T):
            best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            break
    
    if best_result:
        return best_result
=======
def construct():
    deadline = time.time() + 110
    
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)
    
    # Try ranks from 45 to 51 with multiple restarts
    best_result = None
    best_rank = 999
    
    for target_rank in range(45, 52):
        if time.time() > deadline - 15:
            break
        
        # Multiple restarts for each rank
        max_restarts = 10 if target_rank <= 47 else 5
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = 123 + restart * 17
            U, V, W = _als_decompose(T, target_rank, n_iter=150, seed=seed)
            
            # Polish if close
            T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < 1e-6:
                U, V, W = _polish_decomposition(U, V, W, T, n_iter=30)
            
            if _verify_decomposition(U, V, W, T):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = target_rank
                break
        
        if best_result:
            break
    
    if best_result:
        return best_result
>>>>>>> REPLACE
