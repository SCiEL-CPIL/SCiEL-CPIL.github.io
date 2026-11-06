import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer before 150s limit
    
    # Limit for powerful number generation
    # We want to maximize the number of representations a + b = N
    # where a and b are powerful numbers.
    # Powerful numbers up to X are roughly X^(1/3) in count? No, it's related to divisor function.
    # Count of powerful numbers <= x is ~ c * x^(1/3).
    # If we pick N around 2 * limit, the number of representations is roughly the number of powerful numbers
    # that can pair up.
    # Let's estimate the count.
    # For limit = 10^15, max_b = 10^5.
    # Number of powerful numbers is roughly sum_{b squarefree <= 10^5} (10^15 / b^3)^(1/2) = 10^7.5 * sum 1/b^1.5
    # Sum 1/b^1.5 converges.
    # Let's try to push limit as high as possible while keeping generation fast.
    # Generation is O(limit^(2/3)) roughly.
    # For 10^15, limit^(2/3) = 10^10 which is too slow in Python.
    # We need a faster generation or a lower limit.
    # Current limit 3*10^13 takes how long?
    # Let's stick to a limit that generates in < 10s.
    # 10^13 generated in a few seconds in the seed?
    # The seed used 3*10^13.
    
    # Let's optimize the generation loop.
    limit = 10**14
    
    powerful = set([1])
    
    # Max b such that b^3 <= limit
    max_b = int(limit ** (1/3)) + 1
    
    def is_square_free(n):
        if n <= 1:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    # Precompute square-free numbers up to max_b
    # This is faster than checking each b individually if max_b is large,
    # but max_b is ~ 46415 for 10^14.
    # is_square_free is fast enough.
    
    powerful_list = []
    
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b_cubed = b ** 3
        max_a = isqrt(limit // b_cubed)
        for a in range(1, max_a + 1):
            val = a * a * b_cubed
            if val <= limit:
                powerful_list.append(val)
            else:
                break
    
    # Remove duplicates (should be unique by construction a^2 * b^3 with b square-free)
    # a^2 * b^3 = c^2 * d^3 implies same prime factorization exponents.
    # Exponents are 2k + 3m. Unique representation?
    # Yes, powerful numbers have unique representation as a^2 * b^3 with b square-free.
    powerful_list = sorted(set(powerful_list))
    powerful_set = set(powerful_list)
    
    def count_reps(N):
        count = 0
        pairs = []
        # Binary search could be faster for membership, but set is O(1).
        # Iterating through powerful_list is O(len(powerful_list)).
        # len(powerful_list) for 10^14 is approx 10^14^(1/3) * constant?
        # Actually count is ~ c * x^(1/3).
        # For 10^14, x^(1/3) ~ 21544.
        # Wait, the count is much larger.
        # Let's check the size.
        # Sum_{b<=B} sqrt(L/b^3) = sqrt(L) * sum b^(-1.5)
        # Sum b^(-1.5) ~ 2.6
        # sqrt(10^14) = 10^7.
        # So ~ 2.6 * 10^7 powerful numbers.
        # Iterating 2.6 * 10^7 numbers per candidate N is too slow if we check many candidates.
        # We need to be smart about which N to check.
        
        # Strategy:
        # 1. Generate powerful numbers.
        # 2. Find N with max representations.
        # The number of representations r(N) = sum_{a in P} [N-a in P].
        # This is the autocorrelation of the set P.
        # We want to maximize this.
        # 
        # Observation:
        # If we pick N = 2 * M where M is a powerful number, we get at least one pair (M, M).
        # But we want many.
        # 
        # Random search over N in [limit, 2*limit] might find good ones.
        # But 2.6 * 10^7 iterations per check is ~ 0.1s in C, but ~ 1-2s in Python.
        # We can only check ~ 10-20 candidates in 100s.
        # 
        # We need a faster way to find good N.
        # 
        # Alternative:
        # Use FFT? No, N is up to 10^14, array too big.
        # 
        # Heuristic:
        # Sums of powerful numbers are denser around certain values?
        # 
        # Let's try to optimize count_reps.
        # Instead of iterating all powerful numbers, can we do better?
        # No, we need to check each a.
        # 
        # Can we reduce the number of powerful numbers?
        # If we use a smaller limit, we have fewer powerful numbers, but N is smaller.
        # The density of powerful numbers is x^(-2/3).
        # The expected number of representations for a random N is ~ sum_{a} P(N-a in P) ~ sum x^(-2/3) ~ x * x^(-2/3) = x^(1/3).
        # So larger x gives larger expected representations.
        # 
        # Let's try to make the check faster.
        # Use a list of powerful numbers.
        # 
        # We can also try to find N by looking at sums of pairs of powerful numbers.
        # But there are (2.6*10^7)^2 pairs, too many.
        # 
        # Let's stick to random search but make it efficient.
        # 
        # Optimization:
        # If we sort powerful_list, we can break early if a > N/2.
        # 
        # Also, we can pre-compute nothing else really.
        
        # Let's try a different limit to balance generation time and search time.
        # If limit is 10^13, count is ~ 2.6 * 10^6.
        # Check takes ~ 0.1s.
        # We can check 1000 candidates in 100s.
        # 
        # If limit is 10^14, count is ~ 2.6 * 10^7.
        # Check takes ~ 1s.
        # We can check 100 candidates.
        # 
        # Which is better?
        # Expected max in 1000 samples of x^(1/3) vs 100 samples of (10x)^(1/3) = 2.15 * x^(1/3).
        # Max of 1000 samples is likely larger than max of 100 samples even with the factor 2.15.
        # So smaller limit might be better for search.
        
        # Let's restart with limit = 10^13 or 5*10^13.
        # The seed used 3*10^13.
        
        pass

    # Re-evaluating limit
    # Let's use 10^13 for faster checks.
    limit = 10**13
    powerful = set([1])
    max_b = int(limit ** (1/3)) + 1
    
    def is_square_free(n):
        if n <= 1:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    powerful_list = []
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b_cubed = b ** 3
        max_a = isqrt(limit // b_cubed)
        for a in range(1, max_a + 1):
            val = a * a * b_cubed
            if val <= limit:
                powerful_list.append(val)
            else:
                break
    
    powerful_list = sorted(set(powerful_list))
    powerful_set = set(powerful_list)
    
    # print(f"Generated {len(powerful_list)} powerful numbers up to {limit}")
    
    def count_reps(N):
        count = 0
        pairs = []
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        return count, pairs

    best_count = 0
    best_N = 0
    best_pairs = []
    
    # Strategy:
    # 1. Check N = 2 * p for all powerful p? No, too many.
    # 2. Random search in [limit, 2*limit].
    # 3. Local search around best found N.
    
    import random
    
    # Initialize with some random candidates
    candidates = set()
    
    # Add some structured candidates
    # Sums of small powerful numbers?
    # No, we want large N.
    
    # Random sampling
    num_initial = 200
    for _ in range(num_initial):
        N = random.randint(limit, 2 * limit)
        candidates.add(N)
    
    # Also add N = 2 * p for random p
    for _ in range(num_initial):
        p = random.choice(powerful_list)
        if p <= limit:
            candidates.add(2 * p)
    
    # Evaluate candidates
    for N in list(candidates):
        if time.time() > deadline - 10:
            break
        c, p = count_reps(N)
        if c > best_count:
            best_count = c
            best_N = N
            best_pairs = p
    
    # Local search / Hill climbing
    # If we found a good N, try N +/- 1, +/- 2, etc.
    # The function r(N) is not smooth, but maybe nearby values are good?
    
    # Continue with random search if time remains
    while time.time() < deadline - 5:
        # Generate new candidate
        # Mix of random and perturbation of best
        if best_N > 0 and random.random() < 0.5:
            # Perturb best
            delta = random.randint(-100, 100)
            N = best_N + delta
            if N < limit or N > 2 * limit:
                N = random.randint(limit, 2 * limit)
        else:
            N = random.randint(limit, 2 * limit)
        
        if N in candidates:
            continue
            
        c, p = count_reps(N)
        if c > best_count:
            best_count = c
            best_N = N
            best_pairs = p
            
        candidates.add(N)
        
        # If we have very good result, we might want to stop early?
        # No, keep searching.
        
    return {'N': best_N, 'pairs': best_pairs}
