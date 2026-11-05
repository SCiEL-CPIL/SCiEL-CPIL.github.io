import time
import random
from itertools import combinations

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Primes up to 20000
PRIMES = get_primes(20000)
PRIME_SET = set(PRIMES)

def factorize(n):
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

def count_distinct_primes(nums):
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            all_primes.update(factorize(s))
    return len(all_primes)

# Precompute prime factors for all sums up to 20000
MAX_SUM = 20000
prime_factors_cache = [set() for _ in range(MAX_SUM + 1)]
for s in range(2, MAX_SUM + 1):
    prime_factors_cache[s] = factorize(s)

def generate_smooth_numbers(max_val, primes):
    nums = {1}
    for p in primes:
        current = list(nums)
        for x in current:
            y = x
            while y * p <= max_val:
                y *= p
                nums.add(y)
    return sorted(nums)

def construct():
    start_time = time.time()
    deadline = 85.0
    
    # Candidate prime sets to try (small primes first)
    candidate_prime_sets = [
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        [2, 3, 5, 7, 11, 13, 17, 19, 23],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7],
        [2, 3, 5],
        [2, 3],
    ]
    
    best_set = []
    best_score = 100
    
    for prime_set in candidate_prime_sets:
        if time.time() > deadline:
            break
        
        prime_set_set = set(prime_set)
        
        # Find all numbers <= 10000 such that for all pairs (a, b), 
        # a + b has only prime factors from prime_set_set
        # This is expensive, so let's use a different approach:
        # Precompute which numbers can pair with which other numbers
        
        # For each number a, find all numbers b such that a+b is smooth w.r.t. prime_set
        # Then build a graph and find a clique of size 12
        
        # First, identify candidate numbers: those that are "compatible" with many others
        # A number a is compatible with b if all prime factors of (a+b) are in prime_set_set
        
        # To speed this up, let's first filter to numbers where a itself is smooth
        # (though this isn't strictly necessary, it helps)
        
        # Generate smooth numbers
        smooth_nums = generate_smooth_numbers(10000, prime_set)
        
        # For each pair of smooth numbers, check if their sum is smooth
        # Build adjacency: adj[a] = set of b such that a+b is smooth
        
        # To be efficient, only consider the first N smooth numbers
        # where N is large enough to potentially find 12
        max_candidates = min(len(smooth_nums), 200)
        candidates = smooth_nums[:max_candidates]
        
        # Precompute compatibility
        adj = {a: set() for a in candidates}
        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                s = a + b
                if s <= MAX_SUM:
                    pf = prime_factors_cache[s]
                    if pf.issubset(prime_set_set):
                        adj[a].add(b)
                        adj[b].add(a)
        
        # Now find a clique of size 12 in this graph
        # Use backtracking
        
        def find_clique(min_size, candidates_list, adj):
            if len(candidates_list) < min_size:
                return None
            
            # Sort by degree (descending) for better pruning
            candidates_list = sorted(candidates_list, key=lambda x: len(adj[x]), reverse=True)
            
            clique = []
            
            def backtrack(remaining):
                if len(clique) == min_size:
                    return True
                if not remaining:
                    return False
                if len(remaining) < min_size - len(clique):
                    return False
                
                # Try adding each candidate
                for idx, node in enumerate(remaining):
                    # Check if node is connected to all in current clique
                    if all(node in adj[c] for c in clique):
                        clique.append(node)
                        new_remaining = remaining[idx+1:]
                        if backtrack(new_remaining):
                            return True
                        clique.pop()
                
                return False
            
            if backtrack(candidates_list):
                return clique
            return None
        
        clique = find_clique(12, candidates, adj)
        
        if clique:
            score = count_distinct_primes(clique)
            if score < best_score:
                best_score = score
                best_set = clique[:]
                if best_score <= 3:
                    break
        
        # Also try a local search from a random subset
        if time.time() > deadline:
            break
        
        # Random restart with local search
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Start with a random subset of candidates
            if len(candidates) >= 12:
                init_set = random.sample(candidates, 12)
            else:
                init_set = candidates[:]
                while len(init_set) < 12:
                    init_set.append(random.randint(1, 10000))
            
            # Local search: try to replace elements to improve score
            s = init_set[:]
            score = count_distinct_primes(s)
            
            for _ in range(2000):
                if time.time() > deadline:
                    break
                
                i = random.randint(0, 11)
                old_val = s[i]
                new_val = random.choice(candidates)
                if new_val in s:
                    continue
                
                s[i] = new_val
                new_score = count_distinct_primes(s)
                
                if new_score <= score:
                    score = new_score
                else:
                    s[i] = old_val
            
            if score < best_score:
                best_score = score
                best_set = s[:]
                if best_score <= 3:
                    break
    
    # Fallback: if we didn't find anything good, return a reasonable default
    if not best_set:
        smooth_nums = generate_smooth_numbers(10000, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
        best_set = smooth_nums[:12]
    
    return best_set
