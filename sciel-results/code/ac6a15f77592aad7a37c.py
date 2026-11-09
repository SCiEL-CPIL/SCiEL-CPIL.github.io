def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try structured search
    if time.time() < deadline:
        s = _structured_search(deadline)
        if s is not None:
            result = s
            best_rank = RANK_SCHOOLBOOK - 1  # at least one less
    
    # Try random search for lower ranks
    if time.time() < deadline:
        r = _random_search(85, 95, 2000, deadline)
        if r is not None:
            result = r
            best_rank = 85
    
    return result
