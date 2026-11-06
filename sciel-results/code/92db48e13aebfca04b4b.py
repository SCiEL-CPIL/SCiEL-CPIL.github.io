    # Try N from 26 down to 11
    total_time_limit = 115
    start_total = time.time()
    
    # Allocate time based on N
    # Higher N is more important.
    time_allocs = {
        26: 50,
        25: 30,
        24: 20,
        23: 10,
        22: 5
    }
    
    for N in range(26, 10, -1):
        time_remaining = total_time_limit - (time.time() - start_total)
        if time_remaining < 2:
            break
        
        alloc_time = time_allocs.get(N, 5)
        alloc_time = min(alloc_time, time_remaining)
        
        res = solve_backtracking(N, time_limit=alloc_time)
        if res is not None:
            return res
            
    # Fallback to seed solution
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
