import time
import random
from itertools import combinations

def get_primes(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Precompute primes and factorization for sums up to 20000
MAX_SUM = 20000
PRIMES = get_primes(MAX_SUM)
PRIME_SET = set(PRIMES)

def factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            while n % d == 0:
                n //= d
            factors.add(d)
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

# Precompute prime factors for all sums 2..20000
SUM_PRIME_FACTORS = [set() for _ in range(MAX_SUM + 1)]
for s in range(2, MAX_SUM + 1):
    SUM_PRIME_FACTORS[s] = factorize(s)

def count_distinct_primes(nums):
    """Count distinct primes dividing all pairwise sums."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes.update(SUM_PRIME_FACTORS[nums[i] + nums[j]])
    return len(all_primes)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    def time_left():
        return deadline - time.time()
    
    # Strategy: For each candidate prime set S, find 12 numbers whose pairwise sums
    # are all S-smooth. Try increasingly larger S.
    
    # Precompute smooth number sets for common prime sets
    # A number is S-smooth if all its prime factors are in S
    
    def is_smooth(n, prime_set):
        """Check if n is smooth with respect to prime_set."""
        if n <= 1:
            return True
        factors = SUM_PRIME_FACTORS[n] if n <= MAX_SUM else factorize(n)
        return factors.issubset(prime_set)
    
    # Generate candidate numbers: focus on numbers that can form smooth sums
    # Key insight: if we pick numbers that are multiples of small primes,
    # their sums will also have those small primes as factors
    
    # Candidate pool: numbers from 1 to 2000 that are "rich" in small prime factors
    # These are numbers with many small prime factors (smooth numbers)
    def generate_candidate_pool(max_val=3000, max_primes_to_consider=15):
        """Generate candidate numbers that are smooth with respect to first k primes."""
        candidates = set()
        small_primes = PRIMES[:max_primes_to_consider]
        
        # Add all numbers up to 200
        for i in range(1, 201):
            candidates.add(i)
        
        # Add smooth numbers
        def add_smooth(base, remaining_primes, max_val):
            if not remaining_primes:
                return
            p = remaining_primes[0]
            rest = remaining_primes[1:]
            x = base
            while x <= max_val:
                candidates.add(x)
                add_smooth(x, rest, max_val)
                x *= p
        
        add_smooth(1, small_primes, max_val)
        
        # Add some random numbers for diversity
        for _ in range(500):
            candidates.add(random.randint(1, max_val))
        
        return sorted(candidates)
    
    # Try different prime sets in order of increasing size
    # Start with the most restrictive (smallest prime sets)
    
    def try_prime_set(prime_set, candidate_pool, deadline):
        """
        Given a set of allowed primes, find 12 numbers from pool
        where all pairwise sums have prime factors only in prime_set.
        Returns the 12 numbers or None.
        """
        if len(prime_set) < 2:
            return None
        
        # Build compatibility graph
        # Node: number from pool
        # Edge: a-b if prime factors of (a+b) ⊆ prime_set
        
        n_pool = len(candidate_pool)
        adj = [set() for _ in range(n_pool)]
        
        for i in range(n_pool):
            a = candidate_pool[i]
            for j in range(i + 1, n_pool):
                b = candidate_pool[j]
                s = a + b
                if s > MAX_SUM:
                    continue
                pf = SUM_PRIME_FACTORS[s]
                if pf.issubset(prime_set):
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Remove nodes with degree < 11
        valid = [i for i in range(n_pool) if len(adj[i]) >= 11]
        if len(valid) < 12:
            return None
        
        # Find clique of size 12 using backtracking with pruning
        # Order by degree descending
        valid.sort(key=lambda i: -len(adj[i]))
        
        best_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if time.time() > deadline:
                return
            if len(R) >= 12:
                best_clique = R[:]
                return
            if not P:
                return
            
            # Pivot selection
            pivot_candidates = P | X
            if not pivot_candidates:
                return
            pivot = max(pivot_candidates, key=lambda v: len(P & adj[v]))
            
            # Only consider vertices in P that are NOT adjacent to pivot
            candidates = [v for v in P if v not in adj[pivot]]
            
            for v in candidates:
                if time.time() > deadline:
                    return
                R.append(v)
                new_P = P & adj[v]
                new_X = X & adj[v]
                P.remove(v)
                bron_kerbosch(R, new_P, new_X)
                R.pop()
                X.add(v)
                if len(R) >= 12:
                    return
        
        bron_kerbosch([], set(valid), set())
        
        if len(best_clique) >= 12:
            return [candidate_pool[i] for i in best_clique[:12]]
        return None
    
    # Generate candidate prime sets to try
    small_primes = PRIMES[:20]  # First 20 primes
    
    # Size 2: all pairs
    prime_sets_to_try = []
    for i in range(10):
        for j in range(i + 1, 10):
            prime_sets_to_try.append(frozenset([small_primes[i], small_primes[j]]))
    
    # Size 3: all triples of first 8 primes
    for i in range(8):
        for j in range(i + 1, 8):
            for k in range(j + 1, 8):
                prime_sets_to_try.append(frozenset([small_primes[i], small_primes[j], small_primes[k]]))
    
    # Size 4: all quadruples of first 6 primes
    for i in range(6):
        for j in range(i + 1, 6):
            for k in range(j + 1, 6):
                for l in range(k + 1, 6):
                    prime_sets_to_try.append(frozenset([small_primes[i], small_primes[j], small_primes[k], small_primes[l]]))
    
    # Sort by size, then by max prime
    prime_sets_to_try.sort(key=lambda ps: (len(ps), max(ps)))
    
    # Generate candidate pool
    candidate_pool = generate_candidate_pool(3000, 10)
    
    best_result = None
    best_score = float('inf')
    
    # Try each prime set
    for ps in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
        
        clique = try_prime_set(set(ps), candidate_pool, deadline - 3)
        
        if clique is not None and len(clique) >= 12:
            score = count_distinct_primes(clique[:12])
            if score < best_score:
                best_score = score
                best_result = clique[:12]
                if best_score <= 3:
                    break
    
    # If clique search didn't find a good solution, use local search
    if best_result is None or best_score > 5:
        # Generate initial solution
        initial = list(range(1, 13))
        current_score = count_distinct_primes(initial)
        
        if current_score < best_score:
            best_score = current_score
            best_result = initial[:]
        
        # Local search
        def local_search(initial_set, pool, max_iters=5000):
            current = initial_set[:]
            current_score = count_distinct_primes(current)
            best = current[:]
            best_score_local = current_score
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                
                improved = False
                idx = random.randint(0, len(current) - 1)
                old_val = current[idx]
                
                # Try random replacements
                for new_val in random.sample(pool, min(50, len(pool))):
                    if new_val == old_val or new_val in current:
                        continue
                    
                    trial = current[:]
                    trial[idx] = new_val
                    new_score = count_distinct_primes(trial)
                    
                    if new_score < current_score:
                        current = trial
                        current_score = new_score
                        improved = True
                        
                        if new_score < best_score_local:
                            best_score_local = new_score
                            best = trial[:]
                        break
                
                if not improved and it % 100 == 0:
                    # Random restart perturbation
                    idx = random.randint(0, len(current) - 1)
                    new_val = random.choice(pool)
                    if new_val not in current:
                        trial = current[:]
                        trial[idx] = new_val
                        current = trial
                        current_score = count_distinct_primes(current)
            
            return best, best_score_local
        
        refined, refined_score = local_search(best_result, candidate_pool)
        if refined_score < best_score:
            best_score = refined_score
            best_result = refined
    
    if best_result is None:
        best_result = list(range(1, 13))
    
    return best_result
