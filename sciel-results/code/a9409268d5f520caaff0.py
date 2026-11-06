<<<<<<< SEARCH
    # Run a mix of strategies
    strategies = [
        # (window, prefer_small, use_eff, seed)
        (1000, True, True, 1),
        (1000, False, True, 2),
        (5000, True, True, 3),
        (5000, False, True, 4),
        (10000, True, True, 5),
        (10000, False, True, 6),
        (2000, True, True, 7),
        (2000, False, True, 8),
        (500, True, True, 9),
        (500, False, True, 10),
        # Add some non-efficiency strategies
        (1000, True, False, 11),
        (1000, False, False, 12),
        (5000, True, False, 13),
        (5000, False, False, 14),
    ]
    
    # First pass: fixed strategies
    for w, ps, ue, s in strategies:
        if time.time() > deadline - 10:
            break
        run_greedy(w, ps, ue, deadline, s)
    
    # Second pass: v2
    run_greedy_v2(deadline)
    
    # Third pass: v3
    run_greedy_v3(deadline)
    
    # Fourth pass: Randomized perturbations
    if time.time() < deadline - 20:
        for i in range(30):
            if time.time() > deadline - 10:
                break
            w = random.choice([500, 1000, 2000, 5000, 10000])
            ps = random.choice([True, False])
            ue = random.choice([True, False])
            s = 100 + i
            run_greedy(w, ps, ue, deadline, s)
    
    # Fifth pass: Local improvement on best solution
    if best_classes and time.time() < deadline - 5:
        local_improve(best_classes, deadline)
    
    if not best_classes:
        # Fallback
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
    
    return best_classes
=======
    # Run a mix of strategies
    strategies = [
        # (window, prefer_small, use_eff, seed)
        (1000, True, True, 1),
        (1000, False, True, 2),
        (5000, True, True, 3),
        (5000, False, True, 4),
        (10000, True, True, 5),
        (10000, False, True, 6),
        (2000, True, True, 7),
        (2000, False, True, 8),
        (500, True, True, 9),
        (500, False, True, 10),
        # Add some non-efficiency strategies
        (1000, True, False, 11),
        (1000, False, False, 12),
        (5000, True, False, 13),
        (5000, False, False, 14),
    ]
    
    # First pass: fixed strategies
    for w, ps, ue, s in strategies:
        if time.time() > deadline - 10:
            break
        run_greedy(w, ps, ue, deadline, s)
    
    # Second pass: v2
    run_greedy_v2(deadline)
    
    # Third pass: v3
    run_greedy_v3(deadline)
    
    # Fourth pass: Randomized perturbations
    if time.time() < deadline - 20:
        for i in range(30):
            if time.time() > deadline - 10:
                break
            w = random.choice([500, 1000, 2000, 5000, 10000])
            ps = random.choice([True, False])
            ue = random.choice([True, False])
            s = 100 + i
            run_greedy(w, ps, ue, deadline, s)
    
    # Fifth pass: Local improvement on best solution
    if best_classes and time.time() < deadline - 5:
        local_improve(best_classes, deadline)
    
    # Sixth pass: Try more local improvements with different starting points
    if best_classes and time.time() < deadline - 3:
        # Try local improvement again in case we found a better solution
        local_improve(best_classes, deadline)
    
    if not best_classes:
        # Fallback
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
    
    return best_classes
>>>>>>> REPLACE
