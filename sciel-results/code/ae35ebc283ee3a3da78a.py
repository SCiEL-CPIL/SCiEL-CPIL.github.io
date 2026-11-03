<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Start with schoolbook as guaranteed valid solution
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_rank = n * m * p  # 80
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try the structured combined decomposition
    U_comb, V_comb, W_comb = _combine_decompositions()
    if _verify(U_comb, V_comb, W_comb):
        rank_comb = len(U_comb[0])
        if rank_comb < best_rank:
            best_rank = rank_comb
            best_solution = {"U": U_comb, "V": V_comb, "W": W_comb}
    
    # If we have time, try to improve with random search on lower ranks
    if time.time() < deadline - 10:
        for r in range(max(10, best_rank - 10), best_rank):
            if time.time() > deadline - 5:
                break
            
            remaining = deadline - time.time() - 5
            if remaining <= 0:
                break
            
            num_trials = min(500, int(remaining * 100))
            num_trials = max(num_trials, 20)
            
            found = False
            for trial in range(num_trials):
                if time.time() > deadline - 5:
                    break
                
                entries = (-1, 0, 1)
                U = [[random.choice(entries) for _ in range(r)] for _ in range(NM)]
                V = [[random.choice(entries) for _ in range(r)] for _ in range(MP)]
                W = [[random.choice(entries) for _ in range(r)] for _ in range(PN)]
                
                if _verify(U, V, W, tol=1e-9):
                    best_rank = r
                    best_solution = {"U": U, "V": V, "W": W}
                    found = True
                    break
            
            if found:
                continue
    
    return best_solution
=======
def _verify_fast(U, V, W, tol=1e-9):
    """Fast verification using numpy vectorization."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    r = U_np.shape[1]
    
    U_r = U_np.reshape(n, m, r)
    V_r = V_np.reshape(m, p, r)
    W_r = W_np.reshape(p, n, r)
    
    UV = np.einsum('ijc,jkc->ijkr', U_r, V_r)
    tensor = np.einsum('ijkr,kir->ijk', UV, W_r)
    
    return np.allclose(tensor, np.ones((n, m, p)), atol=tol)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Start with schoolbook as guaranteed valid solution
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_rank = n * m * p  # 80
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try the structured combined decomposition
    U_comb, V_comb, W_comb = _combine_decompositions()
    if _verify_fast(U_comb, V_comb, W_comb):
        rank_comb = len(U_comb[0])
        if rank_comb < best_rank:
            best_rank = rank_comb
            best_solution = {"U": U_comb, "V": V_comb, "W": W_comb}
    
    # Try Winograd-style decomposition
    wino = _winograd_4x4_4x5()
    if wino is not None:
        U_w, V_w, W_w = wino
        if _verify_fast(U_w, V_w, W_w):
            rank_w = len(U_w[0])
            if rank_w < best_rank:
                best_rank = rank_w
                best_solution = {"U": U_w, "V": V_w, "W": W_w}
    
    # If we have time, try to improve with random search on lower ranks
    if time.time() < deadline - 10:
        for r in range(max(10, best_rank - 15), best_rank):
            if time.time() > deadline - 5:
                break
            
            remaining = deadline - time.time() - 5
            if remaining <= 0:
                break
            
            # With fast verification, we can do more trials
            num_trials = min(2000, int(remaining * 200))
            num_trials = max(num_trials, 50)
            
            found = False
            for trial in range(num_trials):
                if time.time() > deadline - 5:
                    break
                
                entries = (-1, 0, 1)
                U = [[random.choice(entries) for _ in range(r)] for _ in range(NM)]
                V = [[random.choice(entries) for _ in range(r)] for _ in range(MP)]
                W = [[random.choice(entries) for _ in range(r)] for _ in range(PN)]
                
                if _verify_fast(U, V, W, tol=1e-9):
                    best_rank = r
                    best_solution = {"U": U, "V": V, "W": W}
                    found = True
                    break
            
            if found:
                continue
    
    return best_solution
>>>>>>> REPLACE
