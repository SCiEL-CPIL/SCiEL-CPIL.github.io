<<<<<<< SEARCH
def construct():
    deadline = time.time() + 135.0  # 15s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _build_target(n, m, p)
    T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # Try ranks from 125 down to 85
    for target_rank in range(125, 84, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_als(target_rank, n, m, p, T_flat, deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            break
    
    # If we found a high rank, try to go lower using the found decomposition as init
    if best_rank > 100:
        for target_rank in range(min(best_rank - 1, 100), 84, -1):
            if time.time() > deadline - 10:
                break
            result = _try_rank_als(target_rank, n, m, p, T_flat, deadline,
                                   init_U=best_result['U'], init_V=best_result['V'], init_W=best_result['W'])
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                print(f"Found valid rank {best_rank}")
                break
    
    return best_result
=======
def construct():
    deadline = time.time() + 135.0  # 15s safety margin
    n, m, p = DIMENSIONS
    
    # Start with Laderman-based block decomposition
    U_lad, V_lad, W_lad = _build_5x5_laderman_based()
    lad_rank = len(U_lad[0])
    best_rank = lad_rank
    best_result = {"U": U_lad, "V": V_lad, "W": W_lad}
    
    # Also keep schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    if lad_rank > n * m * p:
        best_rank = n * m * p
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _build_target(n, m, p)
    T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # Try to reduce rank from current best down
    start_rank = best_rank
    # Try a range of ranks below the current best
    for target_rank in range(start_rank - 1, max(84, start_rank - 30), -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_als(target_rank, n, m, p, T_flat, deadline,
                               init_U=best_result['U'], init_V=best_result['V'], init_W=best_result['W'])
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            break
    
    # If we found a rank, try to go even lower
    if best_rank > 90:
        for target_rank in range(best_rank - 1, 84, -1):
            if time.time() > deadline - 10:
                break
            result = _try_rank_als(target_rank, n, m, p, T_flat, deadline,
                                   init_U=best_result['U'], init_V=best_result['V'], init_W=best_result['W'])
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                print(f"Found valid rank {best_rank}")
                break
    
    return best_result
>>>>>>> REPLACE
