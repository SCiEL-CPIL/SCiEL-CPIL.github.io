import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    # 1. Generate pool of 5-smooth numbers (primes 2, 3, 5) up to 10000
    pool = set()
    for a in range(0, 15):
        v2 = 2 ** a
        if v2 > 10000:
            break
        for b in range(0, 15):
            v23 = v2 * (3 ** b)
            if v23 > 10000:
                break
            for c in range(0, 15):
                val = v23 * (5 ** c)
                if val > 10000:
                    break
                pool.add(val)
    pool = sorted(list(pool))
    pool_size = len(pool)
    
    # Precompute smallest prime factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Map pool values to indices for fast lookup
    val_to_idx = {v: i for i, v in enumerate(pool)}
    
    # Precompute factors for all pairs in the pool
    pair_primes = [[None for _ in range(pool_size)] for _ in range(pool_size)]
    
    for i in range(pool_size):
        vi = pool[i]
        for j in range(i + 1, pool_size):
            vj = pool[j]
            s = vi + vj
            factors = set()
            n = s
            while n > 1:
                p = spf[n]
                factors.add(p)
                while n % p == 0:
                    n //= p
            f_tuple = tuple(factors)
            pair_primes[i][j] = f_tuple
            pair_primes[j][i] = f_tuple
            
    if time.time() > deadline - 10.0:
        return random.sample(pool, 12)
    
    # Strategy: Maximum clique search in compatibility graph
    # For a target prime set S, build graph where edge (i,j) exists if
    # all prime factors of pool[i]+pool[j] are in S
    # Find a clique of size 12
    
    # Start with small prime sets and expand
    # Common primes in sums of smooth numbers
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_score = 999
    best_set = pool[:12]
    
    # Try different target prime set sizes
    for target_size in range(3, min(len(base_primes) + 1, 15)):
        if time.time() > deadline - 5.0:
            break
        
        # Get the most common primes from random samples
        prime_freq = defaultdict(int)
        for _ in range(50):
            sample = random.sample(pool, 20)
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    s = sample[i] + sample[j]
                    n = s
                    while n > 1:
                        p = spf[n]
                        prime_freq[p] += 1
                        while n % p == 0:
                            n //= p
        
        # Get top target_size primes
        top_primes = [p for p, _ in sorted(prime_freq.items(), key=lambda x: -x[1])[:target_size]]
        target_set = set(top_primes)
        
        # Build compatibility graph
        # compat[i][j] = True if all prime factors of pool[i]+pool[j] are in target_set
        compat = [[False] * pool_size for _ in range(pool_size)]
        
        for i in range(pool_size):
            vi = pool[i]
            for j in range(i + 1, pool_size):
                vj = pool[j]
                s = vi + vj
                n = s
                compatible = True
                while n > 1:
                    p = spf[n]
                    if p not in target_set:
                        compatible = False
                        break
                    while n % p == 0:
                        n //= p
                if compatible:
                    compat[i][j] = True
                    compat[j][i] = True
        
        # Find maximum clique using greedy + backtracking
        # Use Bron-Kerbosch with pivot for efficiency
        def find_clique(vertices, size_limit=12):
            """Find a clique of size size_limit using Bron-Kerbosch with pivot"""
            if len(vertices) < size_limit:
                return None
            
            # Order vertices by degree in subgraph
            def degree_in_subgraph(v, verts):
                return sum(1 for u in verts if u != v and compat[v][u])
            
            vertices = list(vertices)
            vertices.sort(key=lambda v: degree_in_subgraph(v, vertices), reverse=True)
            
            # Try to find clique by adding vertices in order
            def extend_clique(current_clique, candidates):
                if len(current_clique) == size_limit:
                    return current_clique[:]
                
                if not candidates:
                    return None
                
                # Pivot selection: choose vertex with max degree in candidates
                pivot = max(candidates, key=lambda v: degree_in_subgraph(v, candidates))
                
                # Only consider candidates not adjacent to pivot
                for v in list(candidates):
                    if len(current_clique) + len(candidates) < size_limit:
                        break
                    if compat[v][pivot]:
                        continue
                    
                    new_clique = current_clique + [v]
                    new_candidates = [u for u in candidates if u > v and compat[v][u]]
                    result = extend_clique(new_clique, new_candidates)
                    if result is not None:
                        return result
                    
                    candidates.remove(v)
                
                return None
            
            result = extend_clique([], vertices)
            if result is not None:
                return result
            return None
        
        # Try to find clique from full pool
        clique = find_clique(range(pool_size), 12)
        
        if clique is not None:
            # Verify and compute score
            clique_set = [pool[i] for i in clique]
            prime_counts = {}
            for i in range(12):
                for j in range(i + 1, 12):
                    primes = pair_primes[clique[i]][clique[j]]
                    for p in primes:
                        if p in prime_counts:
                            prime_counts[p] += 1
                        else:
                            prime_counts[p] = 1
            score = len(prime_counts)
            
            if score < best_score:
                best_score = score
                best_set = clique_set[:]
                
                # If we found a small set, try to improve
                if score <= target_size:
                    # Found a valid solution with exactly target_size primes
                    # Try to polish with local search
                    if time.time() < deadline - 3.0:
                        polished = polish_solution(clique_set, pool, pair_primes, deadline - 2.0)
                        if polished is not None:
                            p_score = compute_score(polished, pool, pair_primes)
                            if p_score < best_score:
                                best_score = p_score
                                best_set = polished[:]
        
        # If we found a good solution, try smaller target sets
        if best_score <= target_size:
            continue
    
    # Final polish with local search
    if time.time() < deadline - 2.0:
        polished = polish_solution(best_set, pool, pair_primes, deadline - 1.0)
        if polished is not None:
            p_score = compute_score(polished, pool, pair_primes)
            if p_score < best_score:
                best_score = p_score
                best_set = polished[:]
    
    return best_set

def compute_score(values, pool, pair_primes):
    """Compute the number of distinct prime factors in all pairwise sums"""
    val_to_idx = {v: i for i, v in enumerate(pool)}
    prime_counts = {}
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            idx_i = val_to_idx[values[i]]
            idx_j = val_to_idx[values[j]]
            primes = pair_primes[idx_i][idx_j]
            for p in primes:
                if p in prime_counts:
                    prime_counts[p] += 1
                else:
                    prime_counts[p] = 1
    return len(prime_counts)

def polish_solution(initial_set, pool, pair_primes, deadline):
    """Local search to improve a solution by single swaps"""
    val_to_idx = {v: i for i, v in enumerate(pool)}
    current = initial_set[:]
    current_score = compute_score(current, pool, pair_primes)
    best = current[:]
    best_score = current_score
    
    pool_size = len(pool)
    
    while time.time() < deadline:
        improved = False
        for idx in range(12):
            old_val = current[idx]
            old_idx = val_to_idx[old_val]
            
            best_new_idx = -1
            best_new_score = current_score
            
            for new_idx in range(pool_size):
                new_val = pool[new_idx]
                if new_val in current:
                    continue
                
                # Compute score change
                temp_set = current[:]
                temp_set[idx] = new_val
                candidate_score = compute_score(temp_set, pool, pair_primes)
                
                if candidate_score < best_new_score:
                    best_new_score = candidate_score
                    best_new_idx = new_idx
            
            if best_new_idx != -1 and best_new_score < current_score:
                current[idx] = pool[best_new_idx]
                current_score = best_new_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best = current[:]
        
        if not improved:
            break
    
    return best if best_score < compute_score(initial_set, pool, pair_primes) else None

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# EVOLVE-BLOCK-END
