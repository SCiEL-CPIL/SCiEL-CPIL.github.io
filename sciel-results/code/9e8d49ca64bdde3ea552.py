import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Generate primes up to 1000
    def get_primes(n):
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n+1) if sieve[i]]

    primes = get_primes(1000)
    n_primes = len(primes)  # 168
    p_list = np.array(primes, dtype=np.int64)

    # Precompute coverage contributions for all possible residues for each prime
    # This is a big memory optimization: for prime p, there are p residues,
    # each covering ~limit/p positions. Total = sum(limit/p for p in primes) ~ limit * log(log(1000))
    # For limit=100000, that's roughly 100000 * 3 = 300000 positions total, very manageable.

    def compute_coverage(residues, limit):
        """Compute coverage for x in 1..limit."""
        cov = np.zeros(limit + 1, dtype=np.int32)
        for i, p in enumerate(p_list):
            r = int(residues[i])
            if r == 0:
                start = p
            else:
                start = r
            if start <= limit:
                cov[start::p] += 1
        return cov

    def score_from_coverage(cov, limit):
        """Find largest L such that cov[x] >= 2 for all x in 1..L."""
        bad_indices = np.where(cov[1:limit+1] < 2)[0]
        if len(bad_indices) == 0:
            return limit
        return int(bad_indices[0])

    def full_score(residues, limit=100000):
        cov = compute_coverage(residues, limit)
        return score_from_coverage(cov, limit)

    # Precompute for each prime i and each possible residue r, the set of positions it covers
    # as a boolean array or list of indices. For efficiency, we'll precompute the start position.
    
    # Strategy: Simulated Annealing with a smooth surrogate objective
    # The objective is to maximize the minimum coverage over the prefix.
    # We use a soft-min surrogate: f(residues) = -log(sum(exp(-cov[x]/T))) over x in 1..L
    # But computing this for all x is expensive. Instead, we focus on the bottleneck region.
    
    # Alternative: Use a vectorized coordinate descent with parallel evaluation of candidates.
    # For each prime, we can try multiple candidate residues simultaneously and pick the best.
    
    # Let's implement a fast local search:
    # 1. Start with a greedy initialization
    # 2. Use parallel coordinate descent: for each prime, try all residues and pick the one
    #    that maximizes a score surrogate
    # 3. Use a larger limit for the score computation
    
    CONSTRUCT_LIMIT = 50000
    
    # Phase 1: Greedy construction
    residues = np.zeros(n_primes, dtype=np.int32)
    cov = np.zeros(CONSTRUCT_LIMIT + 1, dtype=np.int32)
    
    for i in range(n_primes):
        p = int(p_list[i])
        best_r = 0
        best_score = -1
        
        # Try all residues for small primes, subset for large
        if p <= 200:
            candidates = range(p)
        else:
            # Focus on residues that cover early numbers
            # The first uncovered number
            bad = np.where(cov[1:CONSTRUCT_LIMIT+1] < 2)[0]
            if len(bad) > 0:
                bottleneck_x = int(bad[0]) + 1
                r_bottleneck = bottleneck_x % p
                candidates = [r_bottleneck]
                for _ in range(50):
                    candidates.append(random.randint(0, p-1))
            else:
                candidates = [random.randint(0, p-1) for _ in range(50)]
        
        for r in candidates:
            if r == 0:
                start = p
            else:
                start = r
            if start > CONSTRUCT_LIMIT:
                benefit = 0
            else:
                positions = np.arange(start, CONSTRUCT_LIMIT+1, p)
                benefit = np.sum(cov[positions] == 1)
                waste = np.sum(cov[positions] >= 2)
                net_benefit = benefit - 0.1 * waste
            
            if net_benefit > best_score:
                best_score = net_benefit
                best_r = r
        
        residues[i] = best_r
        r = best_r
        if r == 0:
            start = p
        else:
            start = r
        if start <= CONSTRUCT_LIMIT:
            cov[start::p] += 1
    
    best_residues = residues.copy()
    cov = compute_coverage(best_residues, CONSTRUCT_LIMIT)
    best_score = score_from_coverage(cov, CONSTRUCT_LIMIT)

    # Phase 2: Parallel coordinate descent with vectorized candidate evaluation
    # For each prime, evaluate all possible residues in parallel and pick the best
    # Use a score based on the number of newly doubly-covered points minus loss
    
    SEARCH_LIMIT = 80000
    
    for iteration in range(50):
        if time.time() > deadline - 30:
            break
        
        cov = compute_coverage(best_residues, SEARCH_LIMIT)
        current_score = score_from_coverage(cov, SEARCH_LIMIT)
        
        if current_score >= SEARCH_LIMIT:
            break
        
        # For each prime, try to find a better residue
        improved = False
        for i in range(n_primes):
            if time.time() > deadline - 30:
                break
            
            p = int(p_list[i])
            old_r = int(best_residues[i])
            
            # Determine which residues to try
            if p <= 100:
                candidates = list(range(p))
            else:
                # Try residues that cover the current bottleneck
                bad = np.where(cov[1:SEARCH_LIMIT+1] < 2)[0]
                if len(bad) > 0:
                    bottleneck_x = int(bad[0]) + 1
                    r_bottleneck = bottleneck_x % p
                    candidates = [r_bottleneck]
                    for _ in range(100):
                        candidates.append(random.randint(0, p-1))
                else:
                    candidates = [random.randint(0, p-1) for _ in range(100)]
            
            best_r = old_r
            best_gain = 0
            
            # Vectorized evaluation of candidates
            # For each candidate r, compute the gain in coverage
            for r in candidates:
                if r == old_r:
                    continue
                
                if r == 0:
                    new_start = p
                else:
                    new_start = r
                if new_start > SEARCH_LIMIT:
                    continue
                
                new_positions = np.arange(new_start, SEARCH_LIMIT+1, p)
                gain = np.sum(cov[new_positions] == 1)
                
                if old_r == 0:
                    old_start = p
                else:
                    old_start = old_r
                if old_start <= SEARCH_LIMIT:
                    old_positions = np.arange(old_start, SEARCH_LIMIT+1, p)
                    loss = np.sum(cov[old_positions] == 1)
                else:
                    loss = 0
                
                net_gain = gain - loss
                
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_r = r
            
            if best_r != old_r:
                best_residues[i] = best_r
                improved = True
        
        if not improved:
            # Try random restarts
            for _ in range(10):
                i = random.randint(0, n_primes - 1)
                p = int(p_list[i])
                best_residues[i] = random.randint(0, p - 1)
        
        new_cov = compute_coverage(best_residues, SEARCH_LIMIT)
        new_score = score_from_coverage(new_cov, SEARCH_LIMIT)
        
        if new_score > best_score:
            best_score = new_score

    # Final evaluation
    final_score = full_score(best_residues, 100000)
    
    return best_residues.tolist()
