def construct():
    import itertools
    
    n = 5040
    target = 4979
    divisors = [d for d in range(1, n+1) if n % d == 0]
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # Split divisors into two halves for Meet-in-the-Middle
    mid = num_divs // 2
    divs1 = divisors[:mid]
    divs2 = divisors[mid:]
    
    for k in range(1, num_divs + 1):
        # Iterate over possible splits of k into k1 and k2
        for k1 in range(max(0, k - len(divs2)), min(k, len(divs1)) + 1):
            k2 = k - k1
            
            # Generate all subsets of size k1 from divs1
            # Map: sum -> list of values (or indices)
            map1 = {}
            for combo in itertools.combinations(divs1, k1):
                s = sum(combo)
                # We only need one solution per sum to save memory
                if s not in map1:
                    map1[s] = list(combo)
            
            # Generate all subsets of size k2 from divs2
            for combo2 in itertools.combinations(divs2, k2):
                s2 = sum(combo2)
                needed = target - s2
                if needed in map1:
                    # Found a solution
                    return map1[needed] + list(combo2)
    
    # Fallback (should not happen for this instance)
    return []
