import time
import random
from itertools import combinations
from collections import defaultdict

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Primes up to 20000 (max sum is 10000+10000=20000)
PRIMES = get_primes(20000)
PRIME_SET = set(PRIMES)
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}

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
SUM_PRIME_FACTORS = [set() for _ in range(20001)]
for s in range(2, 20001):
    SUM_PRIME_FACTORS[s] = factorize(s)

def count_distinct_primes(nums):
    """Count distinct primes dividing all pairwise sums."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes.update(SUM_PRIME_FACTORS[nums[i] + nums[j]])
    return len(all_primes)

def get_prime_set_for_pair(a, b):
    """Get the set of primes dividing a+b."""
    return SUM_PRIME_FACTORS[a + b]

def find_clique(graph, min_size, deadline):
    """
    Find a clique of size min_size in the graph.
    graph: dict mapping node -> set of neighbors
    Returns list of nodes forming the clique, or None.
    """
    nodes = list(graph.keys())
    if len(nodes) < min_size:
        return None
    
    # Order nodes by degree (descending) for better pruning
    nodes.sort(key=lambda x: -len(graph[x]))
    
    # Bron-Kerbosch with pivot
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if len(R) >= min_size:
            best_clique = R[:]
            return
        if not P:
            return
        
        # Pivot selection: choose vertex in P∪X with max degree in P
        pivot_candidates = P | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(P & graph[v]))
        
        # Only consider vertices in P that are NOT adjacent to pivot
        candidates = [v for v in P if v not in graph[pivot]]
        
        for v in candidates:
            if time.time() > deadline:
                return
            R.append(v)
            new_P = P & graph[v]
            new_X = X & graph[v]
            P.remove(v)
            bron_kerbosch(R, new_P, new_X)
            R.pop()
            X.add(v)
            if len(R) >= min_size:
                return
    
    bron_kerbosch([], set(nodes), set())
    return best_clique if len(best_clique) >= min_size else None

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Strategy: Find smallest prime set S such that we can find 12 numbers
    # where all pairwise sums have prime factors only in S.
    
    # Start with small prime sets and expand
    small_primes = PRIMES[:20]  # First 20 primes
    
    # Generate candidate numbers: focus on small numbers and smooth numbers
    # Numbers 1-500 are good candidates since their sums are <= 1000
    candidate_numbers = list(range(1, 501))
    
    # Also add some larger smooth numbers
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
    
    smooth_nums = generate_smooth_numbers(10000, [2, 3, 5, 7])
    for num in smooth_nums:
        if num not in candidate_numbers:
            candidate_numbers.append(num)
    
    candidate_numbers = list(set(candidate_numbers))
    candidate_numbers.sort()
    
    # Try increasing sizes of prime sets
    best_set = None
    best_score = float('inf')
    
    # First, try with very small prime sets
    for prime_count in range(2, 15):
        if time.time() > deadline - 5:
            break
        
        # Try different combinations of small primes
        # Start with the smallest primes
        base_primes = small_primes[:prime_count]
        
        # Build compatibility graph
        # Node: number, Edge: if sum's prime factors are subset of base_primes
        graph = defaultdict(set)
        valid_nodes = set()
        
        for i, a in enumerate(candidate_numbers):
            for b in candidate_numbers[i+1:]:
                s = a + b
                prime_factors = SUM_PRIME_FACTORS[s]
                if prime_factors.issubset(set(base_primes)):
                    graph[a].add(b)
                    graph[b].add(a)
                    valid_nodes.add(a)
                    valid_nodes.add(b)
        
        if len(valid_nodes) < 12:
            continue
        
        # Find clique of size 12
        # Filter graph to only valid nodes
        filtered_graph = {v: graph[v] & valid_nodes for v in valid_nodes}
        
        clique = find_clique(filtered_graph, 12, deadline - 5)
        
        if clique and len(clique) >= 12:
            result = clique[:12]
            score = count_distinct_primes(result)
            if score < best_score:
                best_score = score
                best_set = result
                if score <= prime_count:
                    # Can't do better than using all primes in S
                    break
            break  # Found a solution with this prime set size
    
    # If no clique found, fall back to local search
    if best_set is None or time.time() > deadline - 10:
        # Fallback: use smooth numbers and local search
        smooth_pool = smooth_nums[:100]
        current_set = smooth_pool[:12]
        current_score = count_distinct_primes(current_set)
        
        best_local = current_set[:]
        best_local_score = current_score
        
        def optimize(set_to_opt):
            s = set_to_opt[:]
            score = count_distinct_primes(s)
            for _ in range(2000):
                if time.time() > deadline - 2:
                    break
                i = random.randint(0, 11)
                old_val = s[i]
                new_val = random.choice(smooth_pool)
                if new_val in s:
                    continue
                s[i] = new_val
                new_score = count_distinct_primes(s)
                if new_score <= score:
                    score = new_score
                else:
                    s[i] = old_val
            return s, score
        
        for restart in range(5):
            if time.time() > deadline - 2:
                break
            init_set = random.sample(smooth_pool, 12)
            s, score = optimize(init_set)
            if score < best_local_score:
                best_local_score = score
                best_local = s
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_set = best_local
    
    if best_set is None:
        # Last resort: return first 12 smooth numbers
        best_set = smooth_nums[:12]
    
    return best_set
