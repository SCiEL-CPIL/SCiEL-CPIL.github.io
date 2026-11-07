import math
import time

def construct():
    n = 256
    
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Calculate standalone value for each prime
    standalone_val = {}
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        standalone_val[p] = val
        
    # Calculate merged values and gains for pairs
    # Only consider pairs where product of smallest powers might be <= n
    # p * q <= n is a necessary condition for any merge to be possible
    # Actually p^1 q^1 <= n
    pairs = {}
    for i in range(len(primes)):
        p = primes[i]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            if p * q > n:
                break # Since primes are sorted, q is increasing
            
            # Find max product p^a q^b <= n
            max_prod = 0
            # a can go up to log_p(n)
            max_a = int(math.log(n, p))
            for a in range(1, max_a + 1):
                pa = p ** a
                if pa > n:
                    break
                # b can go up to log_q(n/pa)
                max_b = int(math.log(n / pa, q)) if n / pa >= 1 else 0
                for b in range(1, max_b + 1):
                    prod = pa * (q ** b)
                    if prod <= n and prod > max_prod:
                        max_prod = prod
            
            if max_prod > 0:
                gain = max_prod - standalone_val[p] - standalone_val[q]
                if gain > 0:
                    pairs[(p, q)] = (max_prod, gain)
                    
    # Build adjacency list for active primes
    active_primes = set()
    adj = {p: [] for p in primes}
    for (p, q), (val, gain) in pairs.items():
        active_primes.add(p)
        active_primes.add(q)
        adj[p].append((q, gain, val))
        adj[q].append((p, gain, val))
        
    # Solve maximum weight matching on the subgraph of active primes
    # We use a simple recursive search with memoization on the set of available primes
    # Since the graph is small, we can represent the set as a bitmask or frozenset
    # But frozenset of ints is fine.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # To speed up, we can order primes by degree or index
    active_list = sorted(list(active_primes))
    
    # Map prime to index in active_list for bitmasking if needed, 
    # but frozenset is simpler for small N.
    
    from functools import lru_cache
    
    # We need to pass the current set of available primes.
    # To make it hashable and efficient, we can use a tuple of available primes.
    
    # However, the state space might be large if not pruned.
    # Let's try a greedy approach first, then local search?
    # Given the structure, a simple greedy might be optimal.
    # Let's stick to the exact solver with pruning.
    
    # We will use a dictionary to cache results for a given set of available primes.
    # Key: tuple of available primes (sorted)
    # Value: max gain
    
    cache = {}
    
    def solve(available):
        if not available:
            return 0
        if tuple(available) in cache:
            return cache[tuple(available)]
            
        # Pick the first prime in the sorted list
        p = available[0]
        
        best_gain = 0
        
        # Option 1: Do not match p
        best_gain = max(best_gain, solve(available[1:]))
        
        # Option 2: Match p with a neighbor q in available
        for q, gain, val in adj[p]:
            if q in available:
                # Remove p and q from available
                new_available = [x for x in available if x != p and x != q]
                gain_here = gain + solve(new_available)
                if gain_here > best_gain:
                    best_gain = gain_here
                    
        cache[tuple(available)] = best_gain
        return best_gain

    max_gain = solve(active_list)
    
    # Reconstruct the solution
    # We need to know which pairs were chosen.
    # We can modify solve to return the set of pairs.
    
    cache_pairs = {}
    
    def solve_with_pairs(available):
        if not available:
            return 0, []
        key = tuple(available)
        if key in cache_pairs:
            return cache_pairs[key]
            
        p = available[0]
        
        best_gain = -1
        best_pairs = []
        
        # Option 1: Do not match p
        g1, pairs1 = solve_with_pairs(available[1:])
        if g1 > best_gain:
            best_gain = g1
            best_pairs = pairs1
            
        # Option 2: Match p with a neighbor q
        for q, gain, val in adj[p]:
            if q in available:
                new_available = [x for x in available if x != p and x != q]
                g2, pairs2 = solve_with_pairs(new_available)
                total_gain = gain + g2
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_pairs = pairs2 + [(p, q, val)]
                    
        cache_pairs[key] = (best_gain, best_pairs)
        return best_gain, best_pairs

    _, chosen_pairs = solve_with_pairs(active_list)
    
    # Construct the final list
    result = [1]
    used_primes = set()
    
    for p, q, val in chosen_pairs:
        result.append(val)
        used_primes.add(p)
        used_primes.add(q)
        
    for p in primes:
        if p not in used_primes:
            result.append(standalone_val[p])
            
    # Ensure types are int
    result = [int(x) for x in result]
    
    return result
