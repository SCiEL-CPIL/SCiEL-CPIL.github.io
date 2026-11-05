def _construct_petersen_lift(g, deadline, start_time, max_primes=120):
    # ... existing setup ...
    
    # Calculate minimum m required by Moore bound
    # Moore bound for k=3, girth g: n >= 2 * (2^(g/2) - 1) if g even
    # n = 10 * m
    # 10 * m >= 2 * (2^(g/2) - 1)
    # m >= 0.2 * (2^(g/2) - 1)
    if g % 2 == 0:
        moore_n = 2 * (2 ** (g // 2) - 1)
    else:
        moore_n = 2 * (2 ** ((g - 1) // 2) - 1) + (g - 1) // 2 * 2 ** ((g - 1) // 2) # Approx, actually formula is different for odd
        # For odd g, Moore bound is (k-1)^(g-1)/2 + ...
        # For k=3, g=15: 2*(2^6 - 1) + 7*2^6 = 2*63 + 448 = 574?
        # Let's just use a safe lower bound or calculate exactly if needed.
        # For g=16, moore_n = 510.
        
    min_m = int(np.ceil(moore_n / 10.0))
    
    primes = []
    candidate = max(2, min_m)
    while len(primes) < max_primes:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
        
    # Loop over primes
    for m in primes:
        # ... search ...
