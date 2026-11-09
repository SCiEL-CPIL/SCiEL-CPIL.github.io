import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Generate all 7-smooth numbers up to 20000
    smooth_numbers = set()
    def gen_smooth(n, primes):
        if not primes:
            smooth_numbers.add(n)
            return
        p = primes[0]
        rest = primes[1:]
        m = n
        while m <= 20000:
            gen_smooth(m, rest)
            m *= p
    gen_smooth(1, [7, 5, 3, 2])
    
    # 2. Pre-compute adjacency lists? 
    # Storing full adjacency for 10000 nodes is 100M entries, too much memory/time.
    # Instead, we compute "neighbors" on the fly using the smooth set.
    # But for backtracking, we need to intersect neighbor sets.
    # Let's store neighbors as sets for the top candidates only?
    # Or just use the smooth set check.
    
    def is_smooth_sum(a, b):
        return (a + b) in smooth_numbers

    # We want to find a clique of size 12.
    # Let's use a backtracking search.
    # To make it efficient, we will maintain a list of valid candidates.
    # At each step, we pick a candidate, and filter the remaining candidates to those connected to the picked one.
    
    # Initial candidates: all numbers 1..10000.
    # But 10000 is large. Let's restrict to numbers that have at least some smooth sums.
    # Actually, almost all numbers have some smooth sums.
    # Let's try to pick the "best" starting node.
    
    # Calculate degree for each node to guide the search.
    # degree[v] = number of u in 1..10000 such that v+u is smooth.
    degrees = [0] * 10001
    for s in smooth_numbers:
        start_a = max(1, s - 10000)
        end_a = min(10000, s - 1)
        if start_a > end_a:
            continue
        for a in range(start_a, end_a + 1):
            b = s - a
            if a != b:
                degrees[a] += 1
                degrees[b] += 1
            else:
                degrees[a] += 1 

    # Sort nodes by degree descending
    nodes = list(range(1, 10001))
    nodes.sort(key=lambda x: degrees[x], reverse=True)
    
    # We will try to find the clique starting from the top nodes.
    # To avoid deep recursion and timeouts, we limit the candidate pool at each step.
    # However, the candidate pool shrinks rapidly.
    
    target_size = 12
    
    def find_clique(clique, candidates):
        if time.time() > deadline:
            return None
        if len(clique) == target_size:
            return clique[:]
        
        if not candidates:
            return None
            
        # Pruning: if not enough candidates left
        if len(clique) + len(candidates) < target_size:
            return None
            
        # Try each candidate in candidates
        # We iterate in the order they are given (sorted by degree initially, but we should re-sort or keep sorted?)
        # The candidates list is already sorted by degree from the initial sort, but after filtering, 
        # relative order is preserved. This is a good heuristic.
        
        for i in range(len(candidates)):
            v = candidates[i]
            
            # New candidates: those in candidates[i+1:] that are connected to v
            # We use candidates[i+1:] to avoid revisiting nodes and ensure we find each clique once (lexicographic)
            new_candidates = []
            for u in candidates[i+1:]:
                if is_smooth_sum(v, u):
                    new_candidates.append(u)
                    
            res = find_clique(clique + [v], new_candidates)
            if res is not None:
                return res
                
        return None

    # Start with top K nodes as initial candidates.
    # If we start with all 10000, the first iteration is O(10000^2) which is 10^8, too slow.
    # So we must restrict the initial pool.
    # Let's try increasing pool sizes.
    
    pool_sizes = [200, 500, 1000, 2000, 5000, 10000]
    
    for pool_size in pool_sizes:
        if time.time() > deadline:
            break
            
        candidate_pool = nodes[:pool_size]
        res = find_clique([], candidate_pool)
        
        if res is not None:
            # Verify
            valid = True
            for i in range(len(res)):
                for j in range(i+1, len(res)):
                    if not is_smooth_sum(res[i], res[j]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return res
                
    # Fallback
    return list(range(1, 13))
# EVOLVE-BLOCK-END
