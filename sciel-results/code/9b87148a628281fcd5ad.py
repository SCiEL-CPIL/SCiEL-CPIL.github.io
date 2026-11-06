import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Precompute Smallest Prime Factor (SPF) up to 20000
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    def count_distinct_primes(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes.update(get_factors(s))
        return len(primes)

    # 2. Try to find a set with very low prime count (2 or 3)
    # Strategy: Find 12 distinct integers k_i such that k_i + k_j are 5-smooth.
    # Then a_i = 6 * k_i will have sums with prime factors subset of {2, 3, 5}.
    
    def is_5_smooth(n):
        while n % 2 == 0:
            n //= 2
        while n % 3 == 0:
            n //= 3
        while n % 5 == 0:
            n //= 5
        return n == 1

    # Generate 5-smooth numbers up to 4000
    smooth_5 = set()
    for i in range(1, 4001):
        if is_5_smooth(i):
            smooth_5.add(i)
            
    # Try to find a clique of size 12 in the graph of small integers
    # Vertices 1..200
    # Edge if sum is in smooth_5
    found_set = None
    max_k = 200
    vertices = list(range(1, max_k + 1))
    
    # Precompute adjacency for speed
    adj = {u: set() for u in vertices}
    for u in vertices:
        for v in vertices:
            if u < v and (u + v) in smooth_5:
                adj[u].add(v)
                adj[v].add(u)
                
    # Backtracking search for clique of size 12
    def find_clique(current, candidates, target_size):
        if time.time() > deadline:
            return None
        if len(current) == target_size:
            return list(current)
        if not candidates:
            return None
        
        # Heuristic: pick a vertex with high degree in candidates
        # Or just iterate
        # To be faster, we can prune if len(current) + len(candidates) < target_size
        if len(current) + len(candidates) < target_size:
            return None
            
        for i, v in enumerate(candidates):
            new_current = current + [v]
            # New candidates must be connected to v
            new_candidates = [w for w in candidates[i+1:] if w in adj[v]]
            result = find_clique(new_current, new_candidates, target_size)
            if result:
                return result
        return None

    # Try to find the clique
    # We might need to search a bit. 
    # If we don't find it in the first few seconds, we abort this phase.
    phase_deadline = start_time + 10.0
    
    if time.time() < phase_deadline:
        found_set = find_clique([], vertices, 12)
        
    if found_set:
        # Scale by 6
        result = [6 * k for k in found_set]
        # Verify
        if len(set(result)) == 12 and all(x <= 10000 for x in result):
            return result

    # 3. Fallback: Local Search / Simulated Annealing
    # Start with a random set or a heuristic set
    # Heuristic: Multiples of 6 to keep 2,3 in factors
    # a_i = 6 * random_int
    
    best_score = 999
    best_arr = None
    
    # Initialize with a set of multiples of 6
    # This ensures 2 and 3 are always factors.
    # We want to minimize the *other* primes.
    
    for _ in range(5): # 5 restarts
        arr = [6 * random.randint(1, 1600) for _ in range(12)]
        # Ensure distinct
        while len(set(arr)) < 12:
            arr[random.randint(0, 11)] = 6 * random.randint(1, 1600)
        
        current_score = count_distinct_primes(arr)
        if current_score < best_score:
            best_score = current_score
            best_arr = list(arr)
            
        # Local Search
        temp = 1.0
        for step in range(5000):
            if time.time() > deadline:
                break
                
            i = random.randint(0, 11)
            old_val = arr[i]
            # New value: multiple of 6, distinct
            new_val = 6 * random.randint(1, 1600)
            if new_val == old_val:
                continue
            if new_val in arr:
                # Try to find a distinct one
                for _ in range(10):
                    new_val = 6 * random.randint(1, 1600)
                    if new_val not in arr and new_val != old_val:
                        break
                if new_val in arr or new_val == old_val:
                    continue
                    
            arr[i] = new_val
            new_score = count_distinct_primes(arr)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_arr = list(arr)
                # Cool down
                temp *= 0.99
            else:
                # Accept with probability
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / temp)
                    if random.random() < prob:
                        current_score = new_score
                        temp *= 0.99
                    else:
                        arr[i] = old_val
                else:
                    # Should not happen if we check <, but just in case
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_arr = list(arr)
            
            if current_score == 2: # Theoretical minimum?
                return list(arr)
            if current_score == 3:
                # Try to improve to 2? Unlikely. 
                # But 3 is very good.
                pass

    if best_arr:
        return best_arr
    else:
        # Fallback to baseline
        return list(range(1, 13))
# EVOLVE-BLOCK-END
