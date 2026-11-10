"""Maximize pairwise-noncoprime subset of [1,255] containing 255."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] with gcd(x, 255) > 1
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency as bitmasks for fast intersection checks
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    def compatible(v, clique_mask):
        """Check if v is adjacent to all vertices in clique_mask."""
        return (adj_mask[v] & clique_mask) == clique_mask
    
    def greedy_build(order, forced=0):
        """Build a clique by processing vertices in given order, adding if compatible."""
        clique_mask = 0
        if forced is not None:
            clique_mask |= (1 << forced)
        for v in order:
            if v == forced:
                continue
            if compatible(v, clique_mask):
                clique_mask |= (1 << v)
        return clique_mask
    
    def expand(clique_mask):
        """Try to add any compatible vertices to clique."""
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if (clique_mask >> v) & 1:
                    continue
                if compatible(v, clique_mask):
                    clique_mask |= (1 << v)
                    changed = True
        return clique_mask
    
    best_clique = 0
    best_size = 0
    
    # Strategy 1: Start with multiples of 3 (largest prime group), then expand
    init_mask = 0
    for i, v in enumerate(candidates):
        if v % 3 == 0:
            init_mask |= (1 << i)
    init_mask |= (1 << idx_255)
    best_clique = expand(init_mask)
    best_size = bin(best_clique).count('1')
    
    # Strategy 2: Multiple randomized greedy passes with different orderings
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # Create a randomized ordering biased toward high-degree vertices
        degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
        
        # Shuffle with bias: higher degree vertices earlier
        order = list(range(n))
        random.seed(trial)
        
        # Weighted shuffle: vertices with higher degree appear earlier
        # Use exponential weighting
        weights = [math.exp(degrees[i] / 20.0) for i in range(n)]
        total_w = sum(weights)
        
        # Simple approach: just shuffle, then sort by a random + degree score
        scores = [random.random() + degrees[i] / 50.0 for i in range(n)]
        order.sort(key=lambda i: -scores[i])
        
        # Ensure 255 is first
        order.remove(idx_255)
        order.insert(0, idx_255)
        
        clique = greedy_build(order, forced=idx_255)
        clique = expand(clique)
        size = bin(clique).count('1')
        
        if size > best_size:
            best_clique = clique
            best_size = size
    
    # Strategy 3: Local search with swaps on the best clique
    for _ in range(500):
        if time.time() > deadline - 5:
            break
        
        improved = False
        
        # Try adding any compatible vertex
        for v in range(n):
            if (best_clique >> v) & 1:
                continue
            if compatible(v, best_clique):
                best_clique |= (1 << v)
                best_size += 1
                improved = True
        
        # Try 1-for-1 swaps
        if not improved:
            for rem in range(n):
                if time.time() > deadline - 5:
                    break
                if not ((best_clique >> rem) & 1):
                    continue
                temp = best_clique & ~(1 << rem)
                for add in range(n):
                    if (temp >> add) & 1:
                        continue
                    if compatible(add, temp):
                        new_size = bin(temp | (1 << add)).count('1')
                        if new_size > best_size:
                            best_clique = temp | (1 << add)
                            best_size = new_size
                            improved = True
                            break
                if improved:
                    break
        
        # Try 2-for-1 swaps (remove two, add one) - can help escape local optima
        if not improved:
            clique_list = [i for i in range(n) if (best_clique >> i) & 1]
            for a_idx in range(len(clique_list)):
                if time.time() > deadline - 5:
                    break
                for b_idx in range(a_idx + 1, len(clique_list)):
                    a = clique_list[a_idx]
                    b = clique_list[b_idx]
                    temp = best_clique & ~(1 << a) & ~(1 << b)
                    for add in range(n):
                        if (temp >> add) & 1:
                            continue
                        if compatible(add, temp):
                            new_size = bin(temp | (1 << add)).count('1')
                            if new_size > best_size:
                                best_clique = temp | (1 << add)
                                best_size = new_size
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    # Final polish: aggressive expansion
    best_clique = expand(best_clique)
    best_size = bin(best_clique).count('1')
    
    # Convert back to values
    result = [candidates[i] for i in range(n) if (best_clique >> i) & 1]
    
    # Ensure 255 is present
    if 255 not in result:
        result.append(255)
    
    return sorted(result)
# EVOLVE-BLOCK-END
