<<<<<<< SEARCH
def construct():
    t0 = time.time()
    random.seed(42)  # Reproducibility
    
    best_A = None
    best_peak = float('inf')
    
    # Multiple random restarts with varied parameters
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Alternate between dense removal and sparse completion strategies
        if restart % 2 == 0:
            # Dense removal strategy
            min_size = 25 + (restart % 3) * 5
            max_size = 60 + (restart % 4) * 10
            A = _generate_random_valid_set(min_size=min_size, max_size=max_size)
        else:
            # Sparse completion strategy
            A = _generate_sparse_valid_set()
            if A is None:
                # Fallback to dense if sparse fails
                A = _generate_random_valid_set(min_size=30, max_size=70)
        
        if not _is_valid(A):
            continue
        
        # Run local search with more iterations for early restarts
        remaining = _DEADLINE - time.time() - 10
        if remaining > 20:
            max_iter = 5000
        elif remaining > 10:
            max_iter = 3000
        else:
            max_iter = 1500
            
        A_opt, peak = _local_search(A, max_iterations=max_iter)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A_opt
        
        # Early exit if we found optimal
        if best_peak <= 1:
            break
    
    # If no valid set found, use fallback
    if best_A is None:
        best_A = list(range(64, 129))
        best_peak = _peak(best_A)
    
    return sorted(best_A)
=======
def construct():
    t0 = time.time()
    random.seed(42)  # Reproducibility
    
    best_A = None
    best_peak = float('inf')
    
    # Multiple random restarts with varied parameters
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Alternate between dense removal and sparse completion strategies
        if restart % 2 == 0:
            # Dense removal strategy
            min_size = 25 + (restart % 3) * 5
            max_size = 60 + (restart % 4) * 10
            A = _generate_random_valid_set(min_size=min_size, max_size=max_size)
        else:
            # Sparse completion strategy
            A = _generate_sparse_valid_set()
            if A is None:
                # Fallback to dense if sparse fails
                A = _generate_random_valid_set(min_size=30, max_size=70)
        
        if not _is_valid(A):
            continue
        
        # Run local search with more iterations for early restarts
        remaining = _DEADLINE - time.time() - 10
        if remaining > 25:
            max_iter = 6000
        elif remaining > 15:
            max_iter = 4000
        elif remaining > 10:
            max_iter = 2500
        else:
            max_iter = 1000
            
        A_opt, peak = _local_search(A, max_iterations=max_iter)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A_opt
        
        # Early exit if we found optimal
        if best_peak <= 1:
            break
    
    # If no valid set found, use fallback
    if best_A is None:
        best_A = list(range(64, 129))
        best_peak = _peak(best_A)
    
    return sorted(best_A)
>>>>>>> REPLACE
