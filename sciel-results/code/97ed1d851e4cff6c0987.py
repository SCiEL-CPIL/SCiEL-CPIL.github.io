    if best_result is None:
        candidate = _init_greedy()
    else:
        candidate = best_result
        
    time_left = deadline - time.time()
    if time_left > 10:
        return _sa_optimize(candidate, time_left - 10)
    else:
        return candidate
