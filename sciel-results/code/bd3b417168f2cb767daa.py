import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 80  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Any element must share a factor with 255: divisible by 3, 5, or 17
    # Build candidate pool
    candidates = [n for n in range(1, 256) if math.gcd(n, 255) > 1]
    
    # Precompute gcd relationships
    n_cand = len(candidates)
    # For each candidate, find which other candidates it's compatible with (gcd > 1)
    # We'll use a bitmask or set approach
    
    # Start with the greedy baseline: all multiples of 3
    base_set = set(n for n in candidates if n % 3 == 0)
    best_set = set(base_set)
    
    # Try to improve via local search: try swapping elements or adding
    # First, let's try a smarter greedy: start with 255, then greedily add
    # the candidate with the most connections to current set
    
    # Build adjacency: adj[i] = set of indices j where gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy: start with 255 (find its index)
    idx_255 = candidates.index(255)
    
    # Try multiple greedy strategies
    best_size = len(best_set)
    
    # Strategy 1: All multiples of 3 (already have this, size 85)
    # Strategy 2: Greedy from 255, picking vertex with max degree to current set
    for seed_idx in [idx_255]:
        current = {seed_idx}
        remaining = set(range(n_cand)) - current
        
        while remaining:
            if time.time() > deadline:
                break
            # Find vertex in remaining with max connections to current
            best_v = -1
            best_score = -1
            for v in remaining:
                score = len(adj[v] & current)
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v == -1 or best_score == 0:
                break
            current.add(best_v)
            remaining.remove(best_v)
        
        # Convert to values
        current_vals = set(candidates[i] for i in current)
        if len(current_vals) > best_size:
            best_size = len(current_vals)
            best_set = current_vals
    
    # Strategy 3: Try all multiples of 3, then see if we can swap in better elements
    # Actually, let's try: all numbers divisible by 3 OR (divisible by 5 AND divisible by 3) -- same thing
    # Let's try a different approach: maximum clique via branch and bound on reduced graph
    
    # The key insight: if we pick a "dominant" prime p, all multiples of p form a valid set.
    # Multiples of 3: 85, Multiples of 5: 51, Multiples of 17: 15
    # Can we combine? Only if cross-pairs also have gcd > 1.
    
    # Let's try: all multiples of 3, plus check if any non-multiple-of-3 can be added
    # by removing some conflicting multiples of 3
    # This is a weighted independent set / max clique problem
    
    # For now, return the best we've found
    result = sorted(best_set)
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                # Fallback to safe answer
                return list(range(3, 256, 3))
    
    if 255 not in result:
        result.append(255)
        result.sort()
    
    return result
