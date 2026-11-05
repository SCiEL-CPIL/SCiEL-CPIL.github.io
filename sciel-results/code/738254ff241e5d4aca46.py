import time
import random
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Identify all bad triples
    bad_triples = []
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = a * b // gcd(a, b)
            for c in range(b + 1, n + 1):
                lac = a * c // gcd(a, c)
                if lac != lab:
                    continue
                lbc = b * c // gcd(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return numbers
        
    # 2. Precompute conflict structures
    conflicts = {x: set() for x in numbers}
    # Also precompute for each x, a list of (u, v) pairs for fast degree updates
    triple_neighbors = {x: [] for x in numbers}
    for a, b, c in bad_triples:
        conflicts[a].add(frozenset((b, c)))
        conflicts[b].add(frozenset((a, c)))
        conflicts[c].add(frozenset((a, b)))
        triple_neighbors[a].append((b, c))
        triple_neighbors[b].append((a, c))
        triple_neighbors[c].append((a, b))
    
    def can_add(x, current_set):
        for pair in conflicts[x]:
            if pair.issubset(current_set):
                return False
        return True

    best_set = set()
    best_size = -1
    
    # Precompute: for each pair (u,v), which third vertices form a bad triple with them
    # This helps for swap moves
    pair_to_third = {}
    for a, b, c in bad_triples:
        for p, q in [(a,b), (a,c), (b,c)]:
            if p > q:
                p, q = q, p
            pair_to_third.setdefault((p, q), set()).add(c if (a,b,c) != (p,q,c) else (a if (a,b) != (p,q) else b))
    
    # More careful: for pair (p,q), find all r such that (p,q,r) is a bad triple
    pair_to_third = {}
    for a, b, c in bad_triples:
        for p, q, r in [(a,b,c), (a,c,b), (b,c,a)]:
            if p < q:
                pair_to_third.setdefault((p, q), set()).add(r)
    
    def local_search(S, max_iters=200):
        """Hill climbing with add and swap moves"""
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            
            # Try single add moves
            addable = []
            for x in numbers:
                if x not in S and can_add(x, S):
                    addable.append(x)
            
            if addable:
                # Pick the one that creates fewest new conflicts
                best_x = None
                best_score = float('inf')
                for x in addable:
                    # Count how many pairs in conflicts[x] are "close" to being in S
                    score = sum(1 for pair in conflicts[x] if len(pair & S) == 1)
                    if score < best_score:
                        best_score = score
                        best_x = x
                S.add(best_x)
                improved = True
                continue
            
            # Try swap moves: remove y, add z
            # For each y in S, for each z not in S, check if removing y allows adding z
            # This is expensive, so limit it
            if len(S) < 120:  # Only try swaps if we're not already large
                found_swap = False
                # Sample some y's and z's
                y_candidates = random.sample(list(S), min(20, len(S)))
                z_candidates = [x for x in numbers if x not in S]
                random.shuffle(z_candidates)
                z_candidates = z_candidates[:20]
                
                for y in y_candidates:
                    for z in z_candidates:
                        # Check if removing y and adding z is valid
                        # After removing y, can we add z?
                        # z can be added if no pair in conflicts[z] is subset of (S - {y})
                        valid = True
                        for pair in conflicts[z]:
                            if pair.issubset(S - {y}):
                                valid = False
                                break
                        if valid:
                            # Also check that removing y doesn't break anything (it won't, since we're removing)
                            # But we need to ensure S - {y} + {z} is valid
                            # Since z can be added to S-{y}, and S-{y} is valid (subset of valid S),
                            # the result is valid
                            S.remove(y)
                            S.add(z)
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    continue
            
            if not improved:
                break
        
        return S

    num_restarts = 300
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        S = set(numbers)
        
        # Compute initial degree
        degree = {x: 0 for x in numbers}
        for a, b, c in bad_triples:
            degree[a] += 1
            degree[b] += 1
            degree[c] += 1
        
        # Greedy removal with faster degree updates
        while True:
            if time.time() > deadline - 2.0:
                break
                
            max_deg = 0
            candidates = []
            for x in S:
                if degree[x] > max_deg:
                    max_deg = degree[x]
                    candidates = [x]
                elif degree[x] == max_deg and max_deg > 0:
                    candidates.append(x)
            
            if not candidates or max_deg == 0:
                break
                
            # Random tie-breaking
            target = random.choice(candidates)
            S.remove(target)
            
            # Fast degree update using precomputed neighbors
            for u, v in triple_neighbors[target]:
                if u in S and v in S:
                    degree[u] -= 1
                    degree[v] -= 1
        
        # Local search with add and swap moves
        S = local_search(S, max_iters=300)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        # Early termination if we reach a high threshold
        if best_size >= 100:
            # Continue a few more iterations to see if we can do better
            pass
    
    if best_size == -1:
        return [1]
        
    return list(best_set)
