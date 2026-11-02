def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try Rank 17 first as it's the most likely improvement
    remaining = deadline - time.time() - 20
    if remaining > 10:
        t17_budget = min(80, remaining)
        result = _try_perturb_rank(n, m, p, 17, timeout=t17_budget, n_trials=200)
        if result is not None:
            return result
            
    # Try Rank 16
    remaining = deadline - time.time() - 15
    if remaining > 5:
        t16_budget = min(15, remaining)
        result = _search_rank_aggressive(n, m, p, 16, timeout=t16_budget, n_trials=40)
        if result is not None:
            return result

    # Try Rank 15
    remaining = deadline - time.time() - 10
    if remaining > 5:
        t15_budget = min(10, remaining)
        result = _search_rank_aggressive(n, m, p, 15, timeout=t15_budget, n_trials=20)
        if result is not None:
            return result

    # Fallback Rank 18 (Fast and Guaranteed)
    return _construct_rank18(n, m, p)
