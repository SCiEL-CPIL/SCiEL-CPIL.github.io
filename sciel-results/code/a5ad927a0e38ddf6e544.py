import math
import time

def construct():
    n = 255
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    primes = [3, 5, 17]
    
    # The base best set is all multiples of the smallest prime factor (3)
    # This gives 85 elements, all sharing factor 3 with each other and with 255.
    best_set = list(range(3, n + 1, 3))
    max_size = len(best_set)
    
    # Try to improve by finding a larger pairwise non-coprime set containing 255.
    # Strategy: For each prime p dividing 255, consider the set of all numbers in [1, 255]
    # that are divisible by p. This is a clique of size floor(255/p).
    # The largest such clique is for p=3, giving 85 elements.
    
    # Can we do better than 85? We need a set S containing 255 such that
    # gcd(a,b) > 1 for all distinct a,b in S, and |S| > 85.
    # 
    # Key observation: Any element x in S must share at least one prime factor with 255,
    # i.e., x must be divisible by 3, 5, or 17.
    # 
    # Let A = multiples of 3, B = multiples of 5, C = multiples of 17.
    # |A| = 85, |B| = 51, |C| = 15.
    # 
    # The set A is already a valid clique. Can we add elements from B\A or C\A?
    # An element b in B\A is divisible by 5 but not by 3. For b to be in a pairwise
    # non-coprime set with all elements of A, we need gcd(b, a) > 1 for all a in A.
    # But take a = 3*2 = 6 (in A). gcd(b, 6) > 1 requires b to share a factor with 6,
    # i.e., b divisible by 2 or 3. Since b is not divisible by 3, b must be divisible by 2.
    # But we also need gcd(b, a') > 1 for all a' in A. Take a' = 3*7 = 21. gcd(b, 21) > 1
    # requires b divisible by 3 or 7. Since b not divisible by 3, b must be divisible by 7.
    # This gets complicated. Let's check if we can add ANY element from B\A to A.
    #
    # Actually, the current approach of just using multiples of 3 is optimal for this family.
    # Let's verify by trying a greedy/local search approach to see if we can exceed 85.
    
    # Build the graph: vertices are numbers in [1,255] that share a factor with 255
    # (i.e., divisible by 3, 5, or 17), plus we must include 255.
    # Edge between u and v if gcd(u,v) > 1.
    # We want the largest clique containing 255.
    
    # For efficiency, let's use a different approach:
    # Since the problem is small (at most 85+51+15 - overlaps ≈ ~100 candidates),
    # we can try to find a larger clique.
    
    # Candidates: all numbers in [1,255] divisible by 3, 5, or 17
    candidates = set()
    for p in primes:
        for x in range(p, n + 1, p):
            candidates.add(x)
    
    candidates = sorted(candidates)
    # 255 must be in the set
    if 255 not in candidates:
        candidates.append(255)
        candidates.sort()
    
    # For a clique containing 255, every other element must have gcd > 1 with 255
    # (which is already true by construction) AND gcd > 1 with every other element in the clique.
    
    # Try a greedy approach: start with {255} and greedily add elements that
    # are compatible with all current elements.
    # But greedy may not be optimal. Let's try a more systematic approach.
    
    # Since we have limited time, let's try: for each prime p, the set of multiples of p
    # is a clique. The best single-prime clique is multiples of 3 (size 85).
    # 
    # Can we combine? E.g., take all multiples of 3, and try to add some multiples of 5
    # that are not multiples of 3 but are compatible with all multiples of 3?
    # As shown above, this is very restrictive.
    
    # Let's do a bounded search: try to find a clique larger than 85 using
    # a greedy + local search approach with a time limit.
    
    # Precompute gcd compatibility
    # For speed, use a numpy-based approach if available, else pure Python
    
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False
    
    # Build compatibility matrix for candidates
    # Two candidates i, j are compatible if gcd(candidates[i], candidates[j]) > 1
    m = len(candidates)
    
    if has_numpy and m < 200:
        # Use numpy for faster gcd computation
        cand_arr = np.array(candidates)
        # Compute gcd matrix
        # This is O(m^2) which for m~100 is 10000 ops, fine
        compat = np.zeros((m, m), dtype=bool)
        for i in range(m):
            for j in range(i + 1, m):
                if math.gcd(candidates[i], candidates[j]) > 1:
                    compat[i, j] = True
                    compat[j, i] = True
        np.fill_diagonal(compat, True)
    else:
        compat = [[False] * m for _ in range(m)]
        for i in range(m):
            compat[i][i] = True
            for j in range(i + 1, m):
                if math.gcd(candidates[i], candidates[j]) > 1:
                    compat[i][j] = True
                    compat[j][i] = True
    
    # Find the index of 255
    idx_255 = candidates.index(255)
    
    # Greedy: start with {255}, repeatedly add the vertex with highest degree
    # among those compatible with all current vertices.
    
    def greedy_clique(containing_idx, compat, m, time_limit=5.0):
        start_time = time.time()
        current = {containing_idx}
        
        # Candidates that are compatible with all in current
        def get_compatible(current):
            compatible = []
            for i in range(m):
                if i in current:
                    continue
                ok = True
                for j in current:
                    if not compat[i][j]:
                        ok = False
                        break
                if ok:
                    compatible.append(i)
            return compatible
        
        compatible = get_compatible(current)
        
        while compatible:
            if time.time() - start_time > time_limit:
                break
            
            # Pick the one with highest degree within compatible set
            best = None
            best_deg = -1
            for i in compatible:
                deg = sum(1 for j in compatible if compat[i][j])
                if deg > best_deg:
                    best_deg = deg
                    best = i
            
            if best is None:
                break
            
            current.add(best)
            compatible = [i for i in compatible if i != best]
            compatible = get_compatible(current)
        
        return current
    
    # Try greedy
    greedy_result = greedy_clique(idx_255, compat, m, time_limit=5.0)
    greedy_size = len(greedy_result)
    
    if greedy_size > max_size:
        best_set = [candidates[i] for i in greedy_result]
        max_size = greedy_size
    
    # Try a few different greedy orderings
    if time.time() < deadline:
        # Try random restarts
        import random
        random.seed(42)
        
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            # Randomize the order in which we consider adding vertices
            current = {idx_255}
            remaining = [i for i in range(m) if i != idx_255 and compat[i][idx_255]]
            random.shuffle(remaining)
            
            added = 0
            for i in remaining:
                if time.time() > deadline - 10:
                    break
                # Check if i is compatible with all in current
                ok = True
                for j in current:
                    if not compat[i][j]:
                        ok = False
                        break
                if ok:
                    current.add(i)
                    added += 1
            
            if len(current) > max_size:
                best_set = [candidates[i] for i in current]
                max_size = len(current)
    
    # Verify the result
    if len(best_set) > 1:
        for i in range(len(best_set)):
            for j in range(i + 1, len(best_set)):
                if math.gcd(best_set[i], best_set[j]) == 1:
                    # Invalid, fall back to multiples of 3
                    return list(range(3, n + 1, 3))
    
    if 255 not in best_set:
        return list(range(3, n + 1, 3))
    
    return best_set
