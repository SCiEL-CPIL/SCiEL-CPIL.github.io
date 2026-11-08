def _construct_full_from_sub(sub_U, sub_V, sub_W, n, m, p):
    n_s = n
    m_s = m // 2
    p_s = p
    rank_s = len(sub_U[0])
    rank = 2 * rank_s
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            if j < m_s:
                for c in range(rank_s):
                    U[i*m + j][c] = sub_U[i*m_s + j][c]
            else:
                j_s = j - m_s
                for c in range(rank_s):
                    U[i*m + j][rank_s + c] = sub_U[i*m_s + j_s][c]
                    
    for j in range(m):
        for k in range(p):
            if j < m_s:
                for c in range(rank_s):
                    V[j*p + k][c] = sub_V[j*p_s + k][c]
            else:
                j_s = j - m_s
                for c in range(rank_s):
                    V[j*p + k][rank_s + c] = sub_V[j_s*p_s + k][c]
                    
    for k in range(p):
        for i in range(n):
            for c in range(rank_s):
                W[k*n + i][c] = sub_W[k*n_s + i][c]
                W[k*n + i][rank_s + c] = sub_W[k*n_s + i][c]
                
    return {"U": U, "V": V, "W": W}

def _solve_sub_problem(n_s, m_s, p_s, time_budget, rng):
    best_rank = n_s * m_s * p_s
    start_time = time.time()
    
    for target_rank in range(10, best_rank):
        if time.time() - start_time > time_budget:
            break
            
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5.0:
            break
            
        attempts = 3
        time_per_attempt = remaining / attempts
        
        for attempt in range(attempts):
            if time.time() - start_time > time_budget:
                break
            res = _try_gradient_descent(n_s, m_s, p_s, target_rank, time_budget=time_per_attempt, rng=rng)
            if res is not None:
                return res
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_result = _make_identity_decomposition(n, m, p)
    best_rank = n * m * p
    
    rng = random.Random(12345)
    
    # Strategy 1: Block decomposition
    sub_n, sub_m, sub_p = 2, 2, 5
    time_budget_sub = 60.0
    
    sub_result = _solve_sub_problem(sub_n, sub_m, sub_p, time_budget_sub, rng)
    
    if sub_result is not None:
        full_result = _construct_full_from_sub(sub_result["U"], sub_result["V"], sub_result["W"], n, m, p)
        if _verify(full_result["U"], full_result["V"], full_result["W"], n, m, p):
            new_rank = len(full_result["U"][0])
            if new_rank < best_rank:
                best_result = full_result
                best_rank = new_rank
    
    # Strategy 2: Original search if time permits
    if time.time() < deadline - 15.0:
        result = _try_structured_patterns(n, m, p, time_budget=20, rng=rng)
        if result is not None:
            if len(result["U"][0]) < best_rank:
                best_result = result
                best_rank = len(result["U"][0])
                
        for target_rank in range(best_rank - 1, max(15, best_rank - 10), -1):
            if time.time() > deadline - 10.0:
                break
            for attempt in range(2):
                if time.time() > deadline - 10.0:
                    break
                result = _try_gradient_descent(n, m, p, target_rank, time_budget=10, rng=rng)
                if result is not None:
                    if len(result["U"][0]) < best_rank:
                        best_result = result
                        best_rank = len(result["U"][0])
                    break
                    
    return best_result
