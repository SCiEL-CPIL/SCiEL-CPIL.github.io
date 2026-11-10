import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = bc / (b+c)
    # Condition: (b+c) divides (b*c)
    
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b:
                    triples.append((a, b, c))
    
    # Build unified conflict list: conflicts[x] = list of (y, z) pairs
    # where {x, y, z} forms a forbidden triple
    conflicts = {i: [] for i in range(1, N + 1)}
    for a, b, c in triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Degree
    degree = [0] * (N + 1)
    for a, b, c in triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a forbidden triple."""
        for (y, z) in conflicts[x]:
            if y in S and z in S:
                return False
        return True

    def is_valid(S):
        """Verify S has no forbidden triple."""
        for x in S:
            for (y, z) in conflicts[x]:
                if y in S and z in S:
                    return False
        return True

    def maximize(S, order):
        """Greedy addition until no more elements can be added."""
        improved = True
        while improved:
            if time.time() > deadline:
                return S
            improved = False
            for x in order:
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
        return S

    def greedy_with_order(order):
        """Construct a set using a specific order."""
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def smart_maximize(S):
        """Repeatedly try to add the most 'compatible' element not in S."""
        improved = True
        while improved:
            if time.time() > deadline - 0.5:
                return S
            improved = False
            # Score each candidate by how many of its conflicts are already in S
            # Prefer elements with fewer conflicts in S
            best_x = None
            best_score = -1
            for x in range(1, N + 1):
                if x in S:
                    continue
                # Count how many conflict pairs are both in S
                conflict_count = 0
                for (y, z) in conflicts[x]:
                    if y in S and z in S:
                        conflict_count += 1
                if conflict_count > 0:
                    continue
                # Score: prefer lower degree, then lower value
                score = -degree[x] * 10 + (N - x)
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x is not None:
                S.add(best_x)
                improved = True
        return S

    def local_search(S, max_iter=200):
        """Hill-climbing: try removing one element and re-adding greedily."""
        best_S = set(S)
        best_size = len(S)
        rng_local = random.Random(hash(frozenset(S)) % (2**32))
        
        for _ in range(max_iter):
            if time.time() > deadline - 1.0:
                break
            if len(best_S) == 0:
                break
            # Try removing each element
            for rem in list(best_S):
                trial = set(best_S)
                trial.remove(rem)
                # Try to add elements back in order of increasing degree
                order = sorted(range(1, N + 1), key=lambda x: (degree[x], x))
                maximize(trial, order)
                if len(trial) > best_size:
                    best_S = trial
                    best_size = len(trial)
                    break
        return best_S

    best_set = set()
    best_size = 0
    rng = random.Random(42)
    
    all_elements = list(range(1, N + 1))
    
    # Multiple diverse initial orderings
    orderings = [
        sorted(all_elements, key=lambda x: degree[x]),           # degree asc
        sorted(all_elements, key=lambda x: -degree[x]),          # degree desc
        all_elements,                                             # ascending
        sorted(all_elements, reverse=True),                       # descending
    ]
    
    for order in orderings:
        if time.time() > deadline - 15:
            break
        S = greedy_with_order(order)
        maximize(S, order)
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
    
    # Random orderings
    for _ in range(8):
        if time.time() > deadline - 10:
            break
        order_rand = list(all_elements)
        rng.shuffle(order_rand)
        Sr = greedy_with_order(order_rand)
        maximize(Sr, order_rand)
        if len(Sr) > best_size:
            best_set = set(Sr)
            best_size = len(Sr)
    
    # ILS with adaptive perturbation
    S = set(best_set)
    
    while time.time() < deadline - 2:
        kick_size = rng.randint(1, 4)
        
        if len(S) > kick_size:
            elements_to_remove = rng.sample(list(S), min(kick_size, len(S)))
            for e in elements_to_remove:
                S.remove(e)
        elif len(S) > 0:
            S.clear()
        
        # Randomized order biased toward low-degree elements
        order_ils = list(range(1, N + 1))
        order_ils.sort(key=lambda x: (degree[x], rng.random()))
        
        maximize(S, order_ils)
        
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
    
    # Final local search polish
    if time.time() < deadline - 1:
        polished = local_search(best_set, max_iter=300)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(polished)
    
    # Final verification
    if not is_valid(best_set):
        best_set = set(range(91, N + 1))
        if not is_valid(best_set):
            best_set = {180}
            
    return list(best_set)

# EVOLVE-BLOCK-END
