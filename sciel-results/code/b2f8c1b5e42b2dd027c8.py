def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    if not _verify(U_sb, V_sb, W_sb, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # The target tensor T_ijk = 1 is rank 1.
    # U_ijk = 1, V_jk = 1, W_ki = 1 gives sum 1.
    # Check if rank 1 works.
    if time.time() < deadline - 5:
        U_1 = [[1.0] for _ in range(n * m)]
        V_1 = [[1.0] for _ in range(m * p)]
        W_1 = [[1.0] for _ in range(p * n)]
        
        if _verify(U_1, V_1, W_1, n, m, p):
            return {"U": U_1, "V": V_1, "W": W_1}

    # If rank 1 didn't work (unexpected), fall back to search
    seeds = list(range(20))
    promising_ranks = list(range(2, 25)) + list(range(25, 45)) + list(range(45, 55)) + list(range(55, 71))
    
    for r in promising_ranks:
        if r >= best_rank:
            continue
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            
            result = _try_rank_fast(n, m, p, r, seed, deadline, max_iter=30)
            if result is not None:
                U_new, V_new, W_new = result
                if _verify(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = r
                    found = True
                    break
        
        if found:
            pass
    
    return best_result
