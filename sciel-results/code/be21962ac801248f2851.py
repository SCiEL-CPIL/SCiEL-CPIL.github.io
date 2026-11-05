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
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}

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

# Precompute prime factors for all sums 2..20000
# Using a list of frozensets for fast lookup and issubset
SUM_PRIME_FACTORS = [frozenset() for _ in range(20001)]
for s in range(2, 20001):
    SUM_PRIME_FACTORS[s] = frozenset(factorize(s))

def count_distinct_primes(nums):
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes.update(SUM_PRIME_FACTORS[nums[i] + nums[j]])
    return len(all_primes)

# Generate candidate numbers
def generate_candidate_pool():
    # Small integers are very promising
    pool = set(range(1, 201))
    
    # Smooth numbers (2,3,5,7-smooth)
    def generate_smooth(max_val, primes):
        nums = {1}
        for p in primes:
            current = list(nums)
            for x in current:
                y = x
                while y * p <= max_val:
                    y *= p
                    nums.add(y)
        return nums
    
    smooth = generate_smooth(10000, [2, 3, 5, 7])
    pool.update(smooth)
    
    # Add some random numbers to break symmetry? 
    # No, stick to structured sets first.
    return sorted(list(pool))

CANDIDATE_POOL = generate_candidate_pool()
# Limit pool size to keep clique search fast? 
# If pool is too large, clique search is slow.
# Let's try to keep it manageable. 200 small + ~100 smooth = ~300 nodes.
# Clique of size 12 in 300 nodes is feasible.

def find_clique_size_12(graph, deadline):
    """
    Find a clique of size 12 in the graph.
    graph: dict mapping node -> set of neighbors
    """
    nodes = list(graph.keys())
    if len(nodes) < 12:
        return None
    
    # Order nodes by degree (descending)
    nodes.sort(key=lambda x: -len(graph[x]))
    
    # Map nodes to indices for faster bitset operations if needed, 
    # but simple set intersection is okay for 300 nodes.
    
    best_clique = []
    
    # Bron-Kerbosch with pivot
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
        pivot = max(pivot_candidates, key=lambda v: len(P & graph[v]))
        
        # Candidates in P not adjacent to pivot
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
            if len(R) >= 12:
                return
    
    bron_kerbosch([], set(nodes), set())
    return best_clique if len(best_clique) >= 12 else None

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    best_set = None
    best_score = float('inf')
    
    # Strategy 1: Systematic search over prime sets S
    # We try to find 12 numbers such that all pairwise sums are S-smooth.
    # If found, the score is <= |S|.
    
    # Generate candidate prime sets S
    # Start with small primes.
    # We will test S = {p1}, {p1, p2}, etc.
    # But checking all combinations is expensive.
    # Instead, we can iterate through subsets of the first K primes.
    
    # Let's try a few specific small prime sets first.
    # {2}, {3}, {2,3}, {2,5}, {3,5}, {2,3,5}, etc.
    
    # A more robust approach: 
    # Iterate through the candidate pool and build a graph for a specific S.
    # If the graph has a clique of 12, we are done.
    
    # Which S to try?
    # The prime factors of sums of small integers (1..100) are likely to be small primes.
    # Let's try S being the set of the first m primes.
    
    # Also, we can just run a generic hill-climbing in parallel or as a fallback.
    
    # Let's prioritize the clique search with small prime sets.
    
    # Define a list of prime sets to try, ordered by "likelihood"
    # 1. Single prime: {2}, {3}, {5}, {7}
    # 2. Two primes: {2,3}, {2,5}, {3,5}, {2,7}, ...
    # 3. Three primes: {2,3,5}, ...
    
    # We will generate these on the fly or precompute a list.
    # Given time constraints, we'll just try the first few combinations of the first 10 primes.
    
    small_primes = PRIMES[:10]
    
    # Generate prime sets to test
    prime_sets_to_test = []
    
    # Size 1
    for p in small_primes:
        prime_sets_to_test.append(frozenset([p]))
        
    # Size 2
    for i in range(len(small_primes)):
        for j in range(i+1, len(small_primes)):
            prime_sets_to_test.append(frozenset([small_primes[i], small_primes[j]]))
            
    # Size 3
    for i in range(len(small_primes)):
        for j in range(i+1, len(small_primes)):
            for k in range(j+1, len(small_primes)):
                prime_sets_to_test.append(frozenset([small_primes[i], small_primes[j], small_primes[k]]))
                
    # Sort by size, then by sum of primes (smaller primes are more likely to divide sums)
    prime_sets_to_test.sort(key=lambda s: (len(s), sum(s)))
    
    # Limit the number of sets to try to avoid timeout
    # Each set requires building a graph and searching for a clique.
    # Building graph for 300 nodes is O(N^2) = 90000 ops. Fast.
    # Clique search is the bottleneck.
    
    max_sets_to_try = 50
    
    for S in prime_sets_to_test[:max_sets_to_try]:
        if time.time() > deadline - 10:
            break
            
        # Build compatibility graph
        # Node: number in CANDIDATE_POOL
        # Edge: (a,b) if SUM_PRIME_FACTORS[a+b] is subset of S
        
        graph = {node: set() for node in CANDIDATE_POOL}
        
        # Optimization: Only consider nodes that are "viable"
        # A node is viable if it can form at least 11 edges? 
        # We don't know that yet. Let's just build the full graph.
        
        for i in range(len(CANDIDATE_POOL)):
            a = CANDIDATE_POOL[i]
            for j in range(i + 1, len(CANDIDATE_POOL)):
                b = CANDIDATE_POOL[j]
                s_val = a + b
                if s_val > 20000:
                    continue
                if SUM_PRIME_FACTORS[s_val].issubset(S):
                    graph[a].add(b)
                    graph[b].add(a)
                    
        # Check if any node has degree < 11? If so, it can't be in a clique of 12.
        # Filter nodes
        valid_nodes = [n for n in graph if len(graph[n]) >= 11]
        
        if len(valid_nodes) < 12:
            continue
            
        # Filter graph to valid nodes
        filtered_graph = {n: graph[n] & set(valid_nodes) for n in valid_nodes}
        
        # Search for clique
        clique = find_clique_size_12(filtered_graph, deadline - 5)
        
        if clique:
            # Verify
            score = count_distinct_primes(clique)
            if score < best_score:
                best_score = score
                best_set = clique
                # If we found a set with score == len(S), it's optimal for this S.
                # Can we do better? Only with a smaller S or different S.
                # Since we are iterating S by size, if we find a clique for S of size k,
                # the score is <= k.
                # If score == k, we might still find a smaller score with a different S of size k?
                # No, the score is the number of distinct primes. If the clique uses only primes in S,
                # the score is a subset of S.
                # If we find a clique for S, we have a valid solution.
                # We can break if best_score is very small (e.g., 2 or 3).
                if best_score <= 3:
                    break
            else:
                # If we found a clique but it's not better, we can still continue
                # to see if a smaller S (not possible as we iterate by size) 
                # or a different S of same size yields a better score?
                # Actually, if we iterate by size, the first clique found for size k
                # has score <= k. 
                # If we find a clique for S1 (size k) with score k, and then S2 (size k) 
                # with score k-1, we'd want S2.
                # So we should continue checking other S of the same size if possible.
                # But to save time, let's just break if we find a very good score.
                pass

    # Strategy 2: Fallback Hill Climbing if no good clique found or time remains
    if best_set is None or time.time() < deadline - 20:
        # Use the current program's hill climbing logic
        # But use the precomputed SUM_PRIME_FACTORS for speed
        
        candidate_pool = CANDIDATE_POOL[:200] # Use a smaller pool for speed
        
        def optimize(set_to_opt):
            s = set_to_opt[:]
            score = count_distinct_primes(s)
            for _ in range(2000):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, 11)
                old_val = s[i]
                new_val = random.choice(candidate_pool)
                if new_val in s:
                    continue
                s[i] = new_val
                new_score = count_distinct_primes(s)
                if new_score <= score:
                    score = new_score
                else:
                    s[i] = old_val
            return s, score

        best_local = None
        best_local_score = float('inf')
        
        # Try a few random restarts
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            init_set = random.sample(candidate_pool, 12)
            s, score = optimize(init_set)
            if score < best_local_score:
                best_local_score = score
                best_local = s
        
        if best_local is not None and best_local_score < best_score:
            best_score = best_local_score
            best_set = best_local

    if best_set is None:
        # Last resort
        best_set = CANDIDATE_POOL[:12]
        
    return best_set
