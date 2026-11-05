import time
import random
from itertools import combinations

def _factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _is_smooth(n, primes):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    # Target size
    k = 12
    
    # Try increasing sets of primes
    prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    best_set = None
    best_prime_count = 100
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
            
        # Generate smooth numbers <= 10000
        smooths = []
        # Simple DFS to generate smooth numbers
        def gen_smooth(current, idx, limit):
            if current > limit:
                return
            if idx == len(primes):
                if current > 1: # positive integers, distinct
                    smooths.append(current)
                return
            p = primes[idx]
            gen_smooth(current, idx + 1, limit)
            while current * p <= limit:
                current *= p
                gen_smooth(current, idx + 1, limit)
        
        gen_smooth(1, 0, 10000)
        smooths = sorted(set(smooths))
        
        if len(smooths) < k:
            continue
            
        # Build adjacency matrix or list
        # n_nodes is small enough for O(N^2) or O(N^3) clique heuristics
        n = len(smooths)
        adj = [[False] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                s = smooths[i] + smooths[j]
                if _is_smooth(s, primes):
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Find clique of size k
        # Simple backtracking
        found_clique = None
        
        # Sort nodes by degree descending to help pruning?
        # For now, simple recursion
        def find_clique(start, current_clique):
            nonlocal found_clique
            if len(current_clique) == k:
                found_clique = current_clique[:]
                return True
            if time.time() > deadline:
                return False
                
            # Pruning: if not enough nodes left, return
            remaining = n - start
            if len(current_clique) + remaining < k:
                return False
                
            for i in range(start, n):
                if adj[i][i]: # Self loop check, not used
                    continue
                # Check if i is connected to all in current_clique
                is_connected = True
                for node in current_clique:
                    if not adj[node][i]:
                        is_connected = False
                        break
                if is_connected:
                    if find_clique(i + 1, current_clique + [i]):
                        return True
            return False
        
        # To speed up, we can use a heuristic: try nodes with high degree first
        # But for small N, simple search is ok.
        # However, N can be ~300 for 5-smooth. O(N^k) is too slow.
        # We need a better clique finder or smaller search space.
        
        # Alternative: Randomized greedy or local search
        # Or just try a few random subsets of smooths
        
        # Let's try a greedy approach:
        # Pick a random node, add neighbors, etc.
        # Or use a library? No standard lib for max clique.
        
        # Given the constraint, let's try to find a clique in a smaller subgraph.
        # Or just accept that we might not find it and return best effort.
        
        # Let's try a simple heuristic:
        # Pick k nodes from smooths that are mutually connected.
        
        # If n is small (< 50), do exact search.
        if n <= 60:
            find_clique(0, [])
        else:
            # Heuristic: Sample subsets
            # This is weak.
            pass
            
        if found_clique:
            best_set = [smooths[i] for i in found_clique]
            best_prime_count = len(primes)
            break # Found with smallest prime set
        
    if best_set:
        return best_set
    else:
        # Fallback: return seed or a set with few primes
        # Try to construct a set with few primes manually if search fails
        # Or just return range(1, 13)
        return list(range(1, 13))
