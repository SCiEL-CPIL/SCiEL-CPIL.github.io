def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    primes_N = _primes(N)
    
    # Try a few variations
    for attempt in range(10):
        if time.time() > DEADLINE - 10: # Leave 10s buffer
            break
            
        # Variation: Randomly perturb the initial set?
        # Or just use the same set but random tie-breaking in argmin?
        # np.argmin doesn't support random tie-breaking easily.
        
        # Instead, let's try starting with different subsets.
        # Attempt 0: All
        # Attempt 1: Odd numbers
        # Attempt 2: Numbers not divisible by 3
        # etc.
        
        if attempt == 0:
            points_arr = np.array(list(range(N + 1)))
        elif attempt == 1:
            points_arr = np.array(list(range(1, N + 1, 2))) # Odds
        elif attempt == 2:
            # Remove multiples of 3 from start
            mask = np.mod(np.arange(N+1), 3) != 0
            points_arr = np.array(list(range(N+1)))[mask]
        else:
            # Random subset of size N/2
            points_arr = np.random.choice(N+1, size=N//2, replace=False)
            points_arr = np.sort(points_arr)
            
        current_len = len(points_arr)
        
        for p in primes_N:
            if p > current_len:
                break
            if time.time() > DEADLINE - 5:
                break
                
            residues = np.mod(points_arr, p)
            counts = np.bincount(residues, minlength=p)
            
            if np.any(counts == 0):
                continue
                
            min_class = np.argmin(counts)
            mask = residues != min_class
            points_arr = points_arr[mask]
            current_len = len(points_arr)
            
        if len(points_arr) > best_size:
            best_size = len(points_arr)
            best_set = points_arr.tolist()
            
    return best_set
