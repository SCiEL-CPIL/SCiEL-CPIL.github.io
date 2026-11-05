import time
import random
import math

MAX_VAL = 20000
N_MAX = 10000
K = 12

# Sieve for smallest prime factor
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(get_prime_factors(s))
    return len(primes)

def generate_smooth_numbers(primes, limit):
    """Generate all P-smooth numbers up to limit using BFS."""
    if not primes:
        return {1}
    result = set([1])
    queue = [1]
    while queue:
        v = queue.pop(0)
        for p in primes:
            nv = v * p
            if nv <= limit and nv not in result:
                result.add(nv)
                queue.append(nv)
    return result

def build_adjacency_bitset(smooth_set):
    """Build adjacency as list of frozensets for fast intersection."""
    # adj[x] = set of y != x such that x+y is in smooth_set
    adj = [set() for _ in range(N_MAX + 1)]
    for s in smooth_set:
        if s < 3:  # min sum is 1+2=3
            continue
        if s > 2 * N_MAX:
            continue
        lo = max(1, s - N_MAX)
        hi = min(N_MAX, s - 1)
        for x in range(lo, hi + 1):
            y = s - x
            if x != y and 1 <= y <= N_MAX:
                adj[x].add(y)
    return adj

def find_clique_bitset(adj, target_size, deadline, max_nodes_to_consider=2000):
    """Find a clique of target_size using greedy + backtracking with bitset-style pruning."""
    # Sort nodes by degree descending
    nodes_by_degree = sorted(range(1, N_MAX + 1), key=lambda x: len(adj[x]), reverse=True)
    
    # Limit search to top nodes
    candidates_pool = nodes_by_degree[:max_nodes_to_consider]
    
    # Precompute: for each node in pool, its neighbors also in pool
    pool_set = set(candidates_pool)
    pool_adj = {}
    for v in candidates_pool:
        pool_adj[v] = adj[v] & pool_set
    
    # Greedy initial attempt
    def greedy_clique(start_nodes):
        clique = []
        remaining = set(start_nodes)
        while len(clique) < target_size:
            if not remaining:
                return None
            # Pick node with highest degree in remaining
            best_v = max(remaining, key=lambda x: len(pool_adj[x] & remaining))
            clique.append(best_v)
            remaining = remaining & pool_adj[best_v]
            remaining.discard(best_v)
        return clique
    
    # Try greedy from several random subsets
    for attempt in range(50):
        if time.time() > deadline:
            return None
        subset = random.sample(candidates_pool, min(200, len(candidates_pool)))
        result = greedy_clique(subset)
        if result and len(result) == target_size:
            # Verify
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    if result[j] not in adj[result[i]]:
                        break
                else:
                    continue
                break
            else:
                return result
    
    # Backtracking search with ordering
    def backtrack(candidates, clique, depth):
        if len(clique) == target_size:
            return clique[:]
        if time.time() > deadline:
            return None
        if len(candidates) < target_size - len(clique):
            return None
        
        # Order candidates by degree in candidates
        ordered = sorted(candidates, key=lambda x: len(pool_adj[x] & candidates), reverse=True)
        
        for v in ordered:
            new_clique = clique + [v]
            new_candidates = candidates & pool_adj[v]
            new_candidates.discard(v)
            result = backtrack(new_candidates, new_clique, depth + 1)
            if result:
                return result
        return None
    
    # Try backtracking from top-degree nodes
    for start_count in [100, 200, 500]:
        if time.time() > deadline:
            break
        start_set = set(candidates_pool[:start_count])
        result = backtrack(start_set, [], 0)
        if result:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit
    
    # Start with small prime sets and expand
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_result = None
    best_cost = float('inf')
    
    # Try prime sets of increasing size
    for size in range(2, 15):
        if time.time() > deadline - 3:
            break
        
        P = base_primes[:size]
        
        # Generate P-smooth numbers up to 2*N_MAX
        smooth_set = generate_smooth_numbers(P, 2 * N_MAX)
        
        if len(smooth_set) < 10:
            continue
        
        # Build adjacency
        adj = build_adjacency_bitset(smooth_set)
        
        # Find clique
        clique = find_clique_bitset(adj, K, deadline - 1)
        
        if clique:
            # Verify
            cost = count_distinct_primes(clique)
            if cost <= size:
                if cost < best_cost:
                    best_cost = cost
                    best_result = clique
                return clique  # Found valid solution with this prime set
    
    # If no exact smooth solution found, use simulated annealing
    if time.time() > deadline - 5:
        if best_result:
            return best_result
        return [3 + 6 * k for k in range(12)]
    
    # Simulated annealing with better initialization
    # Try to initialize with numbers that have smooth pairwise sums
    def sa_init():
        """Initialize with numbers whose pairwise sums have few distinct primes."""
        # Use powers of 2 and 3 as starting points
        candidates = []
        for a in range(1, 15):
            for b in range(1, 15):
                v = (2 ** a) * (3 ** b)
                if 1 <= v <= N_MAX:
                    candidates.append(v)
        candidates = list(set(candidates))
        random.shuffle(candidates)
        
        # Greedily pick 12 numbers minimizing distinct primes in sums
        arr = []
        primes_seen = set()
        for v in candidates:
            if len(arr) >= 12:
                break
            # Check if adding v introduces too many new primes
            new_primes = set()
            for u in arr:
                new_primes.update(get_prime_factors(u + v))
            if len(new_primes - primes_seen) <= 3:
                arr.append(v)
                primes_seen.update(new_primes)
        
        while len(arr) < 12:
            arr.append(random.randint(1, N_MAX))
        
        # Ensure distinctness
        seen = set()
        final = []
        for v in arr:
            if v not in seen:
                seen.add(v)
                final.append(v)
        while len(final) < 12:
            v = random.randint(1, N_MAX)
            if v not in seen:
                seen.add(v)
                final.append(v)
        return final[:12]
    
    best_arr = sa_init()
    best_cost = count_distinct_primes(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 3.0
    cooling_rate = 0.9995
    
    # Precompute prime factor masks for all possible sums? Too large.
    # Instead, cache factor computations.
    factor_cache = {}
    
    def cached_factors(n):
        if n not in factor_cache:
            factor_cache[n] = get_prime_factors(n)
        return factor_cache[n]
    
    while time.time() < deadline - 1:
        if T < 0.005:
            break
        
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        
        # Try multiple perturbation strategies
        strategy = random.random()
        if strategy < 0.4:
            new_val = random.randint(1, N_MAX)
        elif strategy < 0.7:
            # Perturb by small amount
            delta = random.choice([-2, -1, 1, 2, 3, 5, 6, 7, 11, 13])
            new_val = old_val + delta
            if new_val < 1 or new_val > N_MAX:
                new_val = random.randint(1, N_MAX)
        else:
            # Try a smooth number
            a = random.randint(0, 13)
            b = random.randint(0, 13)
            c = random.randint(0, 8)
            new_val = (2 ** a) * (3 ** b) * (5 ** c)
            if new_val < 1 or new_val > N_MAX:
                new_val = random.randint(1, N_MAX)
        
        if new_val in current_arr and new_val != old_val:
            continue
        
        current_arr[idx] = new_val
        new_cost = count_distinct_primes(current_arr)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = current_arr[:]
        else:
            delta_cost = new_cost - current_cost
            if delta_cost == 0:
                current_cost = new_cost
            elif random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
            else:
                current_arr[idx] = old_val
        
        T *= cooling_rate
    
    return best_arr
