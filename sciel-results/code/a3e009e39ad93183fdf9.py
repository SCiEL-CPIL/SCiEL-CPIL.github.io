import time
import random
from collections import defaultdict

def _get_prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _generate_smooth_numbers(max_val, primes):
    """Generate all numbers <= max_val whose prime factors are subset of primes."""
    smooth = set()
    n_primes = len(primes)
    
    def rec(idx, current_val):
        if idx == n_primes:
            if current_val <= max_val:
                smooth.add(current_val)
            return
        p = primes[idx]
        val = current_val
        while val <= max_val:
            rec(idx + 1, val)
            if val > max_val // p:
                break
            val *= p
    
    rec(0, 1)
    return smooth

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n_needed = 12
    max_val = 10000
    
    # Strategy: Greedy clique construction with adaptive prime set
    # 1. Start with a small set of "target primes" for the sums
    # 2. Build a compatibility graph where edge (i,j) exists if prime_factors(a_i + a_j) ⊆ target_primes
    # 3. Greedily find a clique of size 12
    # 4. If no clique found, expand the target prime set and retry
    
    # Precompute prime factors for all numbers up to 20000
    max_sum = 2 * max_val
    prime_factors_cache = {}
    for i in range(2, max_sum + 1):
        if i not in prime_factors_cache:
            prime_factors_cache[i] = _get_prime_factors(i)
    
    # Try with increasing sets of target primes
    # Start with the smallest primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    
    best_result = None
    best_prime_count = float('inf')
    
    for num_primes in range(2, len(small_primes) + 1):
        if time.time() > deadline - 5:
            break
        
        target_primes = set(small_primes[:num_primes])
        
        # Generate candidate pool: smooth numbers (with respect to target_primes) up to max_val
        # But also include some non-smooth numbers to increase diversity
        smooth_pool = _generate_smooth_numbers(max_val, sorted(target_primes))
        
        # Build compatibility graph
        # Node i is compatible with node j if prime_factors(pool[i] + pool[j]) ⊆ target_primes
        candidates = sorted(smooth_pool)
        n_candidates = len(candidates)
        
        if n_candidates < n_needed:
            continue
        
        # Build adjacency list
        # adj[i] = set of j > i such that i and j are compatible
        adj = [set() for _ in range(n_candidates)]
        
        for i in range(n_candidates):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, n_candidates):
                s = candidates[i] + candidates[j]
                if s in prime_factors_cache:
                    pf = prime_factors_cache[s]
                    if pf.issubset(target_primes):
                        adj[i].add(j)
                        adj[j].add(i)
        
        # Greedy clique construction
        # Try multiple starting points
        found_clique = None
        
        # Sort candidates by degree (number of compatible neighbors)
        degrees = [len(adj[i]) for i in range(n_candidates)]
        order = sorted(range(n_candidates), key=lambda x: -degrees[x])
        
        # Try greedy from highest-degree nodes
        for start_node in order[:min(50, n_candidates)]:
            if time.time() > deadline - 2:
                break
            
            clique = [start_node]
            available = adj[start_node].copy()
            
            while len(clique) < n_needed and available:
                # Pick the node in available with highest degree within available
                best_next = None
                best_deg = -1
                for node in available:
                    deg_in_available = len(adj[node] & available)
                    if deg_in_available > best_deg:
                        best_deg = deg_in_available
                        best_next = node
                
                if best_next is None:
                    break
                
                clique.append(best_next)
                available = available & adj[best_next]
            
            if len(clique) >= n_needed:
                found_clique = clique[:n_needed]
                break
        
        if found_clique:
            result = [candidates[i] for i in found_clique]
            # Verify
            prime_union = set()
            for i in range(n_needed):
                for j in range(i + 1, n_needed):
                    s = result[i] + result[j]
                    prime_union.update(prime_factors_cache[s])
            
            if len(prime_union) <= num_primes:
                best_result = result
                best_prime_count = len(prime_union)
                # We can stop early if we found a good solution
                if num_primes <= 3:
                    break
            else:
                # The clique construction was valid for the target primes,
                # but let's double-check
                best_result = result
                best_prime_count = len(prime_union)
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: Local search on a broader pool
    random.seed(42)
    
    # Use 2,3,5-smooth numbers as pool
    pool = sorted(_generate_smooth_numbers(max_val, [2, 3, 5]))
    if len(pool) < n_needed:
        pool = list(range(1, max_val + 1))
    
    # Initial random selection
    current_set = random.sample(pool, n_needed)
    
    def evaluate_set(numbers):
        prime_union = set()
        for i in range(n_needed):
            for j in range(i + 1, n_needed):
                s = numbers[i] + numbers[j]
                prime_union.update(prime_factors_cache[s])
        return len(prime_union)
    
    best_score = float('inf')
    best_set = current_set[:]
    current_score = evaluate_set(current_set)
    if current_score < best_score:
        best_score = current_score
        best_set = current_set[:]
    
    step = 0
    while time.time() < deadline:
        step += 1
        idx = random.randint(0, n_needed - 1)
        old_val = current_set[idx]
        new_val = random.choice(pool)
        if new_val in current_set and new_val != old_val:
            continue
        current_set[idx] = new_val
        new_score = evaluate_set(current_set)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val
    
    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
