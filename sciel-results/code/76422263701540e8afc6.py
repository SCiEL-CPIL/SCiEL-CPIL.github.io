import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b <= N:
                    triples.append((a, b, c))
    
    # Build lookup: for each pair (b,c), what a is forbidden
    # And for each a, what (b,c) pairs are forbidden
    triple_map_a = {}  # a -> set of (b,c)
    triple_map_b = {}  # b -> set of (a,c)
    triple_map_c = {}  # c -> set of (a,b)
    
    # Also build map: element -> list of triple indices containing it
    # This helps in calculating conflict degree
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for idx, (a, b, c) in enumerate(triples):
        if a not in triple_map_a:
            triple_map_a[a] = set()
        triple_map_a[a].add((b, c))
        
        if b not in triple_map_b:
            triple_map_b[b] = set()
        triple_map_b[b].add((a, c))
        
        if c not in triple_map_c:
            triple_map_c[c] = set()
        triple_map_c[c].add((a, b))
        
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)

    def has_triple(S):
        """Check if set S contains any forbidden triple."""
        for a in S:
            if a in triple_map_a:
                for b, c in triple_map_a[a]:
                    if b in S and c in S:
                        return True
        return False
    
    def can_add(S, x):
        """Check if we can add x to S without creating a triple."""
        # x could be the 'a' role
        if x in triple_map_a:
            for b, c in triple_map_a[x]:
                if b in S and c in S:
                    return False
        # x could be the 'b' role
        if x in triple_map_b:
            for a, c in triple_map_b[x]:
                if a in S and c in S:
                    return False
        # x could be the 'c' role
        if x in triple_map_c:
            for a, b in triple_map_c[x]:
                if a in S and b in S:
                    return False
        return True
    
    def greedy_build(order):
        """Build a set by greedily adding elements in the given order."""
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    def local_search_add(S, max_iters=2000):
        """Try to add any missing element repeatedly."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            improved = False
            iters += 1
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
        return S
    
    def swap_search(S, rng, max_iters=50000):
        """Try swapping elements to find better configurations."""
        iters = 0
        while iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            iters += 1
            
            y = rng.choice(sorted(S))
            x = rng.randint(1, N)
            if x == y or x in S:
                continue
            
            # Try swap: remove y, add x
            S_without_y = S - {y}
            if can_add(S_without_y, x):
                S = S_without_y | {x}
        return S

    def strategy_full_remove(rng):
        """Start with all elements, remove conflicts greedily."""
        S = set(range(1, N + 1))
        
        # We need to efficiently find and resolve conflicts.
        # Since N is small, we can just scan triples.
        # To improve, we pick the element in the conflict with the highest "conflict degree".
        
        while True:
            if time.time() >= deadline - 10.0:
                break
                
            conflict_triple = None
            # Find any active conflict
            # We can randomize the order of triples we check to avoid bias
            # But shuffling 500 triples every iteration is expensive.
            # Let's just pick the first one found, but maybe check a few random ones?
            # For simplicity, just scan.
            for t in triples:
                a, b, c = t
                if a in S and b in S and c in S:
                    conflict_triple = t
                    break
            
            if conflict_triple is None:
                break
            
            a, b, c = conflict_triple
            
            # Calculate "degree" for a, b, c in current S
            # Degree = number of other triples (containing this element) that are also fully in S
            
            def get_conflict_degree(x):
                deg = 0
                for idx in elem_to_triples[x]:
                    ta, tb, tc = triples[idx]
                    # Check if the other two are in S
                    # x is in the triple.
                    # The other two are the ones != x
                    others = [ta, tb, tc]
                    others.remove(x)
                    if others[0] in S and others[1] in S:
                        deg += 1
                return deg

            deg_a = get_conflict_degree(a)
            deg_b = get_conflict_degree(b)
            deg_c = get_conflict_degree(c)
            
            # Remove the one with the highest degree (resolves most conflicts)
            # Tie-break randomly
            max_deg = max(deg_a, deg_b, deg_c)
            candidates = []
            if deg_a == max_deg: candidates.append(a)
            if deg_b == max_deg: candidates.append(b)
            if deg_c == max_deg: candidates.append(c)
            
            x_to_remove = rng.choice(candidates)
            S.remove(x_to_remove)
            
        # Local search to add back
        S = local_search_add(S)
        return S

    # Strategy 1: Start with high numbers, add lower ones greedily
    S1 = set(range(91, N + 1))
    for x in range(90, 0, -1):
        if can_add(S1, x):
            S1.add(x)
    S1 = local_search_add(S1)
    
    best = S1
    best_size = len(S1)
    
    # Strategy 2: Try various greedy orderings
    rng = random.Random(42)
    
    # Descending order
    order_desc = list(range(N, 0, -1))
    S2 = greedy_build(order_desc)
    S2 = local_search_add(S2)
    if len(S2) > best_size:
        best = S2
        best_size = len(S2)
    
    # Ascending order
    order_asc = list(range(1, N + 1))
    S3 = greedy_build(order_asc)
    S3 = local_search_add(S3)
    if len(S3) > best_size:
        best = S3
        best_size = len(S3)
        
    # Strategy 3: Full Remove Strategy
    # Run multiple times with different seeds/randomness
    for i in range(50):
        if time.time() >= deadline - 10.0:
            break
        S_try = strategy_full_remove(rng)
        if len(S_try) > best_size:
            best = S_try
            best_size = len(S_try)
            
    # Multiple random orderings
    for iteration in range(200):
        if time.time() >= deadline - 10.0:
            break
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_try = greedy_build(order)
        S_try = local_search_add(S_try, max_iters=500)
        
        if len(S_try) > best_size:
            best = S_try
            best_size = len(S_try)
    
    # Swap search on best solution
    best = swap_search(best, rng, max_iters=30000)
    best = local_search_add(best)
    
    # Try more random restarts with swap
    for iteration in range(50):
        if time.time() >= deadline - 5.0:
            break
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_try = greedy_build(order)
        S_try = local_search_add(S_try, max_iters=300)
        S_try = swap_search(S_try, rng, max_iters=5000)
        S_try = local_search_add(S_try)
        
        if len(S_try) > best_size:
            best = S_try
            best_size = len(S_try)
            
    # Also try Full Remove + Swap
    for iteration in range(20):
        if time.time() >= deadline - 5.0:
            break
        S_try = strategy_full_remove(rng)
        S_try = swap_search(S_try, rng, max_iters=5000)
        S_try = local_search_add(S_try)
        if len(S_try) > best_size:
            best = S_try
            best_size = len(S_try)

    # Final local search
    best = local_search_add(best)
    
    # Verify
    if has_triple(best):
        # Fallback: just use numbers > 90
        best = set(range(91, N + 1))
        
    return list(best)
