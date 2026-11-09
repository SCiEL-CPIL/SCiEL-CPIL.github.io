import math
import time

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number (product of distinct primes)."""
    divisors = [1]
    for p in primes:
        new_divs = [d * p for d in divisors]
        divisors.extend(new_divs)
    divisors.sort()
    return divisors

def _tau_perp_squarefree(primes):
    """Count adjacent coprime pairs among divisors of a squarefree number."""
    divs = _divisors_of_squarefree(primes)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _tau_perp_squarefree_fast(primes):
    """Faster scoring using bitmask representation and precomputed coprimality."""
    k = len(primes)
    if k == 0:
        return 0
    if k == 1:
        return 1
    
    # Generate all divisors as (value, mask) pairs
    # mask bit i is set if prime[i] divides the divisor
    divs = [(1, 0)]
    for i, p in enumerate(primes):
        new_divs = [(d[0] * p, d[1] | (1 << i)) for d in divs]
        divs.extend(new_divs)
    
    # Sort by value
    divs.sort(key=lambda x: x[0])
    
    # Count adjacent coprime pairs: gcd(d_i, d_{i+1}) == 1 iff masks are disjoint
    count = 0
    for i in range(len(divs) - 1):
        if divs[i][1] & divs[i+1][1] == 0:
            count += 1
    return count

def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate primes up to ~2000 for a much wider search space
    all_primes = []
    for x in range(2, 2000):
        if _is_prime(x):
            all_primes.append(x)
    
    best_score = -1
    best_n = 2
    best_primes = [2]
    
    # Score cache to avoid recomputation
    score_cache = {}
    
    def get_score(prime_list):
        """Get score with caching using fast bitmask method."""
        key = tuple(prime_list)
        if key in score_cache:
            return score_cache[key]
        n = 1
        for p in prime_list:
            n *= p
        if n > 10**18:
            score_cache[key] = (-1, n)
            return (-1, n)
        s = _tau_perp_squarefree_fast(prime_list)
        score_cache[key] = (s, n)
        return (s, n)
    
    def try_candidate(prime_list):
        nonlocal best_score, best_n, best_primes
        if time.time() > deadline:
            return False
        s, n = get_score(prime_list)
        if n <= 10**18 and s > best_score:
            best_score = s
            best_n = n
            best_primes = list(prime_list)
        return True
    
    # Phase 1: Try primorials of first k primes for k=2..15
    for k in range(2, 16):
        primes_k = all_primes[:k]
        n_prod = 1
        for p in primes_k:
            n_prod *= p
        if n_prod > 10**18:
            break
        try_candidate(primes_k)
    
    # Phase 2: Local search with single swaps, using cached scores
    def local_search(initial_primes, max_iter=200):
        nonlocal best_score, best_n, best_primes
        current_primes = list(initial_primes)
        current_n = 1
        for p in current_primes:
            current_n *= p
        current_score = _tau_perp_squarefree_fast(current_primes)
        
        # Use a set for O(1) membership check
        current_set = set(current_primes)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return False
            improved = False
            best_candidate = None
            best_candidate_score = current_score
            
            # Try single swaps: remove one prime, add another
            for i in range(len(current_primes)):
                for j in range(len(all_primes)):
                    if all_primes[j] in current_set:
                        continue
                    new_n = current_n // current_primes[i] * all_primes[j]
                    if new_n > 10**18:
                        continue
                    candidate = current_primes[:i] + [all_primes[j]] + current_primes[i+1:]
                    candidate.sort()
                    s, n = get_score(candidate)
                    if n <= 10**18 and s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
            
            if best_candidate is not None:
                current_primes = best_candidate
                current_set = set(current_primes)
                current_n = 1
                for p in current_primes:
                    current_n *= p
                current_score = best_candidate_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_n
                    best_primes = list(current_primes)
            
            if not improved:
                break
        return True
    
    # Phase 3: Multiple local search restarts from diverse starting points
    for start_k in range(10, 16):
        if time.time() > deadline:
            return best_n
        primes_k = all_primes[:start_k]
        n_prod = 1
        for p in primes_k:
            n_prod *= p
        if n_prod <= 10**18:
            local_search(primes_k, max_iter=100)
    
    # Phase 4: Local search from best found
    if time.time() < deadline:
        local_search(best_primes, max_iter=500)
    
    # Phase 5: Try different size sets
    if time.time() < deadline:
        for target_size in range(12, 17):
            if time.time() > deadline:
                return best_n
            base = all_primes[:target_size]
            n_prod = 1
            for p in base:
                n_prod *= p
            if n_prod <= 10**18:
                local_search(base, max_iter=150)
    
    # Phase 6: Final local search polish
    if time.time() < deadline:
        local_search(best_primes, max_iter=300)
    
    return best_n
