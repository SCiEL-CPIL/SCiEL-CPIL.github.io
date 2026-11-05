import time
import random
import math
from itertools import combinations

# Precompute Smallest Prime Factors (SPF) for fast factorization
MAX_VAL = 20000
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

# Precompute factor sets for all sums
SUM_PRIME_FACTORS = [set() for _ in range(MAX_VAL + 1)]
for s in range(2, MAX_VAL + 1):
    SUM_PRIME_FACTORS[s] = get_prime_factors(s)

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(SUM_PRIME_FACTORS[s])
    return len(primes)

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
    deadline = start_time + 80.0  # Safety margin
    
    # Generate candidate pool
    # 1. Small numbers
    candidates = set(range(1, 101))
    # 2. 5-smooth numbers
    smooth_5 = generate_smooth_numbers(10000, [2, 3, 5])
    candidates.update(smooth_5)
    # 3. 7-smooth numbers
    smooth_7 = generate_smooth_numbers(10000, [2, 3, 5, 7])
    candidates.update(smooth_7)
    # 4. 11-smooth numbers (just in case)
    smooth_11 = generate_smooth_numbers(10000, [2, 3, 5, 7, 11])
    candidates.update(smooth_11)
    
    candidate_list = sorted(candidates)
    
    # Helper to check if a set of numbers forms a valid clique for a given prime set
    def find_clique_for_prime_set(candidate_nodes, target_prime_set, min_size, deadline):
        """
        Find a clique of size min_size in the graph where edges exist 
        if the sum's prime factors are a subset of target_prime_set.
        """
        target_set = set(target_prime_set)
        graph = {v: set() for v in candidate_nodes}
        valid_nodes = set()
        
        # Build graph
        for i, a in enumerate(candidate_nodes):
            for b in candidate_nodes[i+1:]:
                s = a + b
                if s > MAX_VAL:
                    continue
                if SUM_PRIME_FACTORS[s].issubset(target_set):
                    graph[a].add(b)
                    graph[b].add(a)
                    valid_nodes.add(a)
                    valid_nodes.add(b)
                    
        if len(valid_nodes) < min_size:
            return None
            
        # Filter graph to valid nodes
        filtered_graph = {v: graph[v] & valid_nodes for v in valid_nodes}
        
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
            
            # Pivot selection
            pivot_candidates = P | X
            if not pivot_candidates:
                return
            pivot = max(pivot_candidates, key=lambda v: len(P & filtered_graph[v]))
            
            candidates = [v for v in P if v not in filtered_graph[pivot]]
            
            for v in candidates:
                if time.time() > deadline:
                    return
                R.append(v)
                new_P = P & filtered_graph[v]
                new_X = X & filtered_graph[v]
                P.remove(v)
                bron_kerbosch(R, new_P, new_X)
                R.pop()
                X.add(v)
                if len(R) >= min_size:
                    return
        
        bron_kerbosch([], set(valid_nodes), set())
        return best_clique if len(best_clique) >= min_size else None

    best_result = None
    best_score = float('inf')
    
    # Try 5-smooth sums (primes 2, 3, 5)
    # This would yield a score of at most 3
    if time.time() < deadline - 20:
        clique = find_clique_for_prime_set(candidate_list, [2, 3, 5], 12, deadline - 20)
        if clique:
            score = count_distinct_primes(clique)
            if score < best_score:
                best_score = score
                best_result = clique
            if best_score <= 3:
                return best_result

    # Try 7-smooth sums (primes 2, 3, 5, 7)
    # This would yield a score of at most 4
    if time.time() < deadline - 15:
        clique = find_clique_for_prime_set(candidate_list, [2, 3, 5, 7], 12, deadline - 15)
        if clique:
            score = count_distinct_primes(clique)
            if score < best_score:
                best_score = score
                best_result = clique
            if best_score <= 4:
                return best_result

    # If no clique found, use Simulated Annealing
    # Start with a random sample from the candidate pool
    if best_result is None:
        best_result = random.sample(candidate_list, 12)
        best_score = count_distinct_primes(best_result)
    
    current_arr = best_result[:]
    current_score = best_score
    
    T = 5.0
    cooling_rate = 0.99
    
    while time.time() < deadline - 2:
        if T < 0.01:
            break
            
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        
        # Pick a new value from the candidate pool
        new_val = random.choice(candidate_list)
        
        if new_val in current_arr and new_val != old_val:
            continue
            
        current_arr[idx] = new_val
        new_score = count_distinct_primes(current_arr)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_result = current_arr[:]
        else:
            delta = new_score - current_score
            if random.random() < math.exp(-delta / T):
                current_score = new_score
            else:
                current_arr[idx] = old_val
        
        T *= cooling_rate
        
    return best_result
