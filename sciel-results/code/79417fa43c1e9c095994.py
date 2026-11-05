import time
import random
from itertools import combinations
from collections import defaultdict

# ---------- primes and factorization ----------

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

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

# Precompute prime factors for all possible sums (2..20000)
SUM_PRIME_FACTORS = [set() for _ in range(20001)]
for s in range(2, 20001):
    SUM_PRIME_FACTORS[s] = factorize(s)

def count_distinct_primes(nums):
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes.update(SUM_PRIME_FACTORS[nums[i] + nums[j]])
    return len(all_primes)

# ---------- candidate pool ----------

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

# Build a focused candidate pool: small integers + smooth numbers
SMOOTH_2357 = generate_smooth_numbers(10000, [2, 3, 5, 7])
SMALL_INTS = list(range(1, 301))
CANDIDATE_POOL = sorted(set(SMALL_INTS + SMOOTH_2357))
# Keep pool size manageable for graph construction
if len(CANDIDATE_POOL) > 800:
    CANDIDATE_POOL = CANDIDATE_POOL[:800]

# ---------- clique finding ----------

def find_clique(graph, min_size, deadline):
    """Bron-Kerbosch with pivot to find a clique of size min_size."""
    nodes = list(graph.keys())
    if len(nodes) < min_size:
        return None
    nodes.sort(key=lambda x: -len(graph[x]))
    
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
        pivot_candidates = P | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(P & graph[v]))
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

# ---------- local search ----------

def local_search(initial_set, pool, deadline, max_iters=3000):
    s = initial_set[:]
    score = count_distinct_primes(s)
    s_set = set(s)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        i = random.randint(0, 11)
        old_val = s[i]
        new_val = random.choice(pool)
        if new_val in s_set:
            continue
        s[i] = new_val
        s_set.discard(old_val)
        s_set.add(new_val)
        new_score = count_distinct_primes(s)
        if new_score <= score:
            score = new_score
        else:
            s[i] = old_val
            s_set.discard(new_val)
            s_set.add(old_val)
    return s, score

# ---------- main construct ----------

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    best_set = None
    best_score = float('inf')
    
    # Strategy: iteratively expand prime set S, build compatibility graph, find 12-clique
    # Start with the smallest primes and grow S by adding the most frequent "new" prime
    
    # Initial S: first few primes
    S = set(PRIMES[:5])  # {2, 3, 5, 7, 11}
    max_prime_set_size = 15
    expansion_rounds = 0
    
    while len(S) < max_prime_set_size and time.time() < deadline - 15 and expansion_rounds < 20:
        expansion_rounds += 1
        
        # Build compatibility graph: edge between a,b if prime_factors(a+b) ⊆ S
        graph = defaultdict(set)
        valid_nodes = set()
        
        for i, a in enumerate(CANDIDATE_POOL):
            for b in CANDIDATE_POOL[i+1:]:
                s = a + b
                pf = SUM_PRIME_FACTORS[s]
                if pf.issubset(S):
                    graph[a].add(b)
                    graph[b].add(a)
                    valid_nodes.add(a)
                    valid_nodes.add(b)
        
        if len(valid_nodes) < 12:
            # Not enough compatible nodes; expand S
            # Find most frequent new prime across incompatible pairs
            new_prime_counts = defaultdict(int)
            for i, a in enumerate(CANDIDATE_POOL):
                for b in CANDIDATE_POOL[i+1:]:
                    s = a + b
                    pf = SUM_PRIME_FACTORS[s]
                    new_pf = pf - S
                    for p in new_pf:
                        new_prime_counts[p] += 1
            if new_prime_counts:
                most_common = max(new_prime_counts, key=new_prime_counts.get)
                S.add(most_common)
            continue
        
        # Find 12-clique in the compatibility graph
        clique_deadline = min(deadline - 15, time.time() + 10)
        clique = find_clique(graph, 12, clique_deadline)
        
        if clique and len(clique) >= 12:
            result = clique[:12]
            score = count_distinct_primes(result)
            if score < best_score:
                best_score = score
                best_set = result
                # If we found a clique using only primes in S, and |S| is small,
                # we can stop early if score is very low
                if score <= len(S):
                    break
            # Even if we found a clique, try to refine with local search
            break
        
        # No clique found; expand S by most frequent new prime
        new_prime_counts = defaultdict(int)
        for i, a in enumerate(CANDIDATE_POOL):
            for b in CANDIDATE_POOL[i+1:]:
                s = a + b
                pf = SUM_PRIME_FACTORS[s]
                new_pf = pf - S
                for p in new_pf:
                    new_prime_counts[p] += 1
        if new_prime_counts:
            most_common = max(new_prime_counts, key=new_prime_counts.get)
            S.add(most_common)
    
    # Local search refinement
    if time.time() < deadline - 5:
        # Refine the best clique found
        if best_set is not None:
            refined, refined_score = local_search(best_set, CANDIDATE_POOL, deadline - 2)
            if refined_score < best_score:
                best_score = refined_score
                best_set = refined
        
        # Try a few random initializations from smooth numbers
        smooth_pool = SMOOTH_2357[:100]
        for restart in range(5):
            if time.time() > deadline - 3:
                break
            init_set = random.sample(smooth_pool, 12)
            s, score = local_search(init_set, smooth_pool, deadline - 2, max_iters=1500)
            if score < best_score:
                best_score = score
                best_set = s
    
    # Fallback
    if best_set is None:
        best_set = SMOOTH_2357[:12]
    
    return best_set
