<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin

    best_S = None
    best_size = 0
    
    # Phase 1: Diverse Greedy Initializations
    # 1. Largest First
    order_largest = list(range(NMAX, 1, -1))
    S1 = _greedy_addition(order_largest, deadline)
    if S1 and _is_valid(S1) and len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
        
    # 2. Smallest First
    order_smallest = list(range(2, NMAX + 1))
    S2 = _greedy_addition(order_smallest, deadline)
    if S2 and _is_valid(S2) and len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
    
    # 4. Randomized Orders
    for seed in [1, 7, 42, 100, 200, 300, 400]:
        if time.time() > deadline - 25:
            break
        rng = random.Random(seed)
        order_rand = list(range(2, NMAX + 1))
        rng.shuffle(order_rand)
        Sr = _greedy_addition(order_rand, deadline)
        if Sr and _is_valid(Sr) and len(Sr) > best_size:
            best_S = Sr
            best_size = len(Sr)
            
    if not best_S:
        best_S = [2, 3] 
        best_size = 2

    # Phase 2: Local Search Improvement with multiple runs
    # Run 1 on best_S
    if time.time() < deadline - 5:
        S_improved = _local_search_improve(best_S, deadline - 5, max_iters=500, rng_seed=42)
        if S_improved and _is_valid(S_improved) and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
            
    # Run 2 on perturbed version
    if time.time() < deadline - 2:
        rng_perturb = random.Random(123)
        S_perturbed = best_S[:]
        for _ in range(rng_perturb.randint(1, 2)):
            if S_perturbed:
                idx = rng_perturb.randrange(len(S_perturbed))
                S_perturbed.pop(idx)
                
        S_improved2 = _local_search_improve(S_perturbed, deadline - 2, max_iters=300, rng_seed=123)
        if S_improved2 and _is_valid(S_improved2) and len(S_improved2) > best_size:
            best_S = S_improved2
            best_size = len(S_improved2)

    # Final Validation
    if not _is_valid(best_S):
        # Fallback to a known valid set if something went wrong
        # Primes are always valid
        primes = [i for i in range(2, NMAX + 1) if all(i % j != 0 for j in range(2, int(i**0.5) + 1))]
        if primes:
            best_S = primes
        else:
            best_S = [2, 3]

    return [int(x) for x in best_S]
=======
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 35s safety margin before 90s kill
    
    # Precompute conflict pairs
    conflict_pairs = _precompute_conflict_pairs()
    
    best_S = None
    best_size = 0
    
    # Phase 1: Greedy Removal with different strategies
    strategies = ["highest_degree", "random_highest", "lowest_value_highest"]
    
    for strat in strategies:
        if time.time() > deadline - 20:
            break
        S = _greedy_removal(conflict_pairs, deadline - 20, strategy=strat)
        if S and _is_valid(S) and len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # Also try a few random restarts of greedy removal
    for seed in [1, 7, 42, 100]:
        if time.time() > deadline - 15:
            break
        random.seed(seed)
        S = _greedy_removal(conflict_pairs, deadline - 15, strategy="random_highest")
        if S and _is_valid(S) and len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    if not best_S:
        # Fallback: primes are always valid
        primes = [i for i in range(2, NMAX + 1) if all(i % j != 0 for j in range(2, int(i**0.5) + 1))]
        best_S = primes
        best_size = len(primes)

    # Phase 2: Local Search Improvement
    if time.time() < deadline - 3:
        S_improved = _local_search_improve(best_S, deadline - 3, conflict_pairs, max_iters=2000, rng_seed=42)
        if S_improved and _is_valid(S_improved) and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
    
    # Run 2 on perturbed version
    if time.time() < deadline - 2:
        rng_perturb = random.Random(123)
        S_perturbed = best_S[:]
        for _ in range(rng_perturb.randint(1, 3)):
            if S_perturbed:
                idx = rng_perturb.randrange(len(S_perturbed))
                S_perturbed.pop(idx)
                
        S_improved2 = _local_search_improve(S_perturbed, deadline - 2, conflict_pairs, max_iters=1000, rng_seed=123)
        if S_improved2 and _is_valid(S_improved2) and len(S_improved2) > best_size:
            best_S = S_improved2
            best_size = len(S_improved2)

    # Final Validation
    if not _is_valid(best_S):
        # Fallback to a known valid set if something went wrong
        # Primes are always valid
        primes = [i for i in range(2, NMAX + 1) if all(i % j != 0 for j in range(2, int(i**0.5) + 1))]
        if primes:
            best_S = primes
        else:
            best_S = [2, 3]

    return [int(x) for x in best_S]
>>>>>>> REPLACE
