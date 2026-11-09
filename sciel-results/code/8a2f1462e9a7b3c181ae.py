<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = []
    best_count = 0
    num_attempts = 20
    
    for i in range(num_attempts):
        if time.time() > deadline:
            break
        result = _greedy_construct(deadline, seed=i * 7919 + 104729)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    if time.time() < deadline - 5:
        best_result = _local_search(best_result, deadline)
    
    return best_result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_result = []
    best_count = 0
    
    # Phase 1: Multiple greedy constructions with varied strategies
    num_greedy = 30
    for i in range(num_greedy):
        if time.time() > deadline - 30:  # Reserve 30s for local search
            break
        strategy = i % 5
        seed = i * 7919 + 104729
        result = _greedy_construct(deadline - 30, seed, strategy)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    # Phase 2: Local search on best result
    if best_result and time.time() < deadline - 5:
        best_result = _local_search(best_result, deadline - 5)
    
    # Phase 3: If time remains, try local search on second-best greedy
    if time.time() < deadline - 10:
        # Quick second pass: try a few more greedy + local search
        for i in range(5):
            if time.time() > deadline - 5:
                break
            strategy = (i + 10) % 5
            seed = (i + 30) * 7919 + 104729
            result = _greedy_construct(deadline - 15, seed, strategy)
            if len(result) > best_count:
                best_count = len(result)
                best_result = result
                if time.time() < deadline - 5:
                    best_result = _local_search(best_result, deadline - 5)
    
    return best_result
>>>>>>> REPLACE
