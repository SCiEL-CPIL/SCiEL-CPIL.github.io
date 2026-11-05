import time
from math import gcd
from functools import lru_cache

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute all bad triples
    # A triple (a,b,c) with a<b<c is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == lab and _lcm(b, c) == lab:
                    bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    
    T = len(bad_triples)
    
    # Precompute for each element, the list of triple indices it appears in
    elem_to_triples = [[] for _ in range(n)]
    for ti, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(ti)
        elem_to_triples[b].append(ti)
        elem_to_triples[c].append(ti)
    
    # For each triple, store the three elements as a bitmask
    triple_mask = [0] * T
    for ti, (a, b, c) in enumerate(bad_triples):
        triple_mask[ti] = (1 << a) | (1 << b) | (1 << c)
    
    # A set S is valid iff for no triple is S a superset of the triple
    # i.e., for all ti, (S & triple_mask[ti]) != triple_mask[ti]
    
    # Greedy: start with all elements, remove to break triples
    # Then local search: try to add back removed elements, or swap
    
    def is_valid(mask):
        """Check if mask represents a valid set"""
        for ti in range(T):
            if (mask & triple_mask[ti]) == triple_mask[ti]:
                return False
        return True
    
    def count_violations(mask):
        """Count how many triples are fully contained in mask"""
        cnt = 0
        for ti in range(T):
            if (mask & triple_mask[ti]) == triple_mask[ti]:
                cnt += 1
        return cnt
    
    def can_add(mask, elem):
        """Check if adding elem to mask keeps it valid"""
        new_mask = mask | (1 << elem)
        # Only need to check triples containing elem
        for ti in elem_to_triples[elem]:
            if (new_mask & triple_mask[ti]) == triple_mask[ti]:
                return False
        return True
    
    # Phase 1: Greedy construction
    # Start with empty set, add elements in order of "least constrained"
    # Sort by degree (number of bad triples they're in), add low-degree first
    
    # Actually, let's try: start with all, greedily remove
    mask = (1 << n) - 1  # all elements
    # Remove elements to break all triples
    # Greedy: repeatedly remove element that breaks most remaining triples
    
    # Better: start empty, add greedily
    mask = 0
    # Try multiple orders
    import random
    rng = random.Random(42)
    
    best_mask = 0
    best_size = 0
    
    # Multiple restarts with different strategies
    strategies = [
        list(range(n)),                          # natural order
        sorted(range(n), key=lambda x: len(elem_to_triples[x])),  # low degree first
        sorted(range(n), key=lambda x: -len(elem_to_triples[x])), # high degree first
    ]
    
    for _ in range(50):
        strategies.append(rng.sample(range(n), n))
    
    for order in strategies:
        if time.time() > deadline:
            break
        mask = 0
        for elem in order:
            if can_add(mask, elem):
                mask |= (1 << elem)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Local search - try to improve
    # Operations: add a removed element, or swap (remove one, add another)
    
    def local_search(mask, max_iters=5000):
        current = mask
        current_size = bin(current).count('1')
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try adding each removed element
            for elem in range(n):
                if not (current & (1 << elem)):
                    if can_add(current, elem):
                        current |= (1 << elem)
                        current_size += 1
                        improved = True
                        break
            # Try swaps: remove one present, add one absent
            if not improved:
                present = [e for e in range(n) if current & (1 << e)]
                absent = [e for e in range(n) if not (current & (1 << e))]
                for rem in present:
                    if time.time() > deadline:
                        return current
                    for add in absent:
                        new_mask = (current & ~(1 << rem)) | (1 << add)
                        if can_add(current & ~(1 << rem), add):
                            # Check full validity
                            valid = True
                            for ti in range(T):
                                if (new_mask & triple_mask[ti]) == triple_mask[ti]:
                                    valid = False
                                    break
                            if valid:
                                current = new_mask
                                improved = True
                                break
                    if improved:
                        break
        return current
    
    best_mask = local_search(best_mask)
    best_size = bin(best_mask).count('1')
    
    # Phase 3: More restarts with local search
    for restart in range(100):
        if time.time() > deadline:
            break
        # Random subset as starting point
        mask = 0
        for e in range(n):
            if rng.random() < 0.5:
                mask |= (1 << e)
        # Fix: remove elements to make valid
        # Greedy removal
        while True:
            if time.time() > deadline:
                break
            # Find a violating triple
            viol = -1
            for ti in range(T):
                if (mask & triple_mask[ti]) == triple_mask[ti]:
                    viol = ti
                    break
            if viol == -1:
                break
            # Remove one element from this triple
            a, b, c = bad_triples[viol]
            # Remove the one with highest degree
            degs = [(len(elem_to_triples[a]), a), (len(elem_to_triples[b]), b), (len(elem_to_triples[c]), c)]
            degs.sort(reverse=True)
            mask &= ~(1 << degs[0][1])
        
        if bin(mask).count('1') > best_size:
            best_mask = local_search(mask)
            best_size = bin(best_mask).count('1')
    
    # Convert to list
    result = [e + 1 for e in range(n) if best_mask & (1 << e)]
    if not result:
        result = [1]
    return result
