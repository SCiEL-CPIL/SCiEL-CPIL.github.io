import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = bc/(b+c), so (b+c) | bc
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b <= N:
                    triples.append((a, b, c))
    
    # Build conflict graph: two elements conflict if they can form a triple with some third element
    # Actually, we need: (x, y) is a conflict pair if there exists z such that {x,y,z} is a forbidden triple
    # But the constraint is on TRIPLES, not pairs. Let me think differently.
    # 
    # For a set S to be valid, no three distinct elements in S can form a triple.
    # So (x,y) together are "conflicting" with z if {x,y,z} is a triple.
    # 
    # For can_add(S, x): x conflicts with S if there exist y,z in S such that {x,y,z} is a triple.
    # This means: for each pair (y,z) in S, if x completes a triple with them, we can't add x.
    
    # Build: for each element x, what pairs (y,z) would complete a triple with x
    triple_map_a = {}  # a -> set of (b,c) where a < b < c
    triple_map_b = {}  # b -> set of (a,c) where a < b < c  
    triple_map_c = {}  # c -> set of (a,b) where a < b < c
    
    for a, b, c in triples:
        if a not in triple_map_a:
            triple_map_a[a] = set()
        triple_map_a[a].add((b, c))
        
        if b not in triple_map_b:
            triple_map_b[b] = set()
        triple_map_b[b].add((a, c))
        
        if c not in triple_map_c:
            triple_map_c[c] = set()
        triple_map_c[c].add((a, b))
    
    # Build conflict degree: for each x, how many other elements y are such that
    # there exists z with {x,y,z} forming a triple. This helps prioritize less-conflicting elements.
    # Actually, let's build: for each x, the set of y such that x and y together conflict with some z
    # i.e., there exists z where {x,y,z} is a forbidden triple
    conflict_with = [set() for _ in range(N + 1)]  # conflict_with[x] = set of y that conflict with x
    
    for a, b, c in triples:
        # a conflicts with b and c (because together with the third, they form a triple)
        conflict_with[a].add(b)
        conflict_with[a].add(c)
        conflict_with[b].add(a)
        conflict_with[b].add(c)
        conflict_with[c].add(a)
        conflict_with[c].add(b)
    
    # Degree of each vertex in conflict graph
    degree = [len(conflict_with[i]) for i in range(N + 1)]
    
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
    
    def removal_search(S, max_iters=1000):
        """Try removing elements that block the most additions, then re-add."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            improved = False
            iters += 1
            # For each element in S, count how many missing elements it blocks
            best_removal = None
            best_gain = 0
            for y in S:
                S_without_y = S - {y}
                # Count how many missing elements can be added
                gain = 0
                for x in range(1, N + 1):
                    if x not in S_without_y:
                        if can_add(S_without_y, x):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_removal = y
            if best_removal is not None and best_gain > 0:
                S = S - {best_removal}
                S = local_search_add(S, max_iters=200)
                improved = True
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
    
    def swap2_search(S, rng, max_iters=20000):
        """Try 2-for-2 swaps to find better configurations."""
        iters = 0
        sorted_S = sorted(S)
        while iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            iters += 1
            
            # Pick two elements to remove
            if len(sorted_S) < 2:
                break
            y1, y2 = rng.sample(sorted_S, 2)
            # Pick two elements to add
            missing = [x for x in range(1, N + 1) if x not in S]
            if len(missing) < 2:
                break
            x1, x2 = rng.sample(missing, 2)
            
            S_removed = S - {y1, y2}
            # Try adding both
            if can_add(S_removed, x1):
                S_tmp = S_removed | {x1}
                if can_add(S_tmp, x2):
                    S = S_tmp | {x2}
                    sorted_S = sorted(S)
            elif can_add(S_removed, x2):
                S_tmp = S_removed | {x2}
                if can_add(S_tmp, x1):
                    S = S_tmp | {x1}
                    sorted_S = sorted(S)
        return S
    
    # Strategy 1: Start with high numbers, add lower ones greedily
    S1 = set(range(91, N + 1))
    for x in range(90, 0, -1):
        if can_add(S1, x):
            S1.add(x)
    S1 = local_search_add(S1)
    S1 = removal_search(S1, max_iters=200)
    S1 = local_search_add(S1)
    
    best = S1
    best_size = len(S1)
    
    # Strategy 2: Degree-based greedy (least conflicts first)
    rng = random.Random(42)
    
    # Sort by degree (ascending) - elements with fewer conflicts are easier to include
    order_by_degree = sorted(range(1, N + 1), key=lambda x: degree[x])
    S4 = greedy_build(order_by_degree)
    S4 = local_search_add(S4)
    S4 = removal_search(S4, max_iters=100)
    S4 = local_search_add(S4)
    if len(S4) > best_size:
        best = S4
        best_size = len(S4)
    
    # Descending order
    order_desc = list(range(N, 0, -1))
    S2 = greedy_build(order_desc)
    S2 = local_search_add(S2)
    S2 = removal_search(S2, max_iters=100)
    S2 = local_search_add(S2)
    if len(S2) > best_size:
        best = S2
        best_size = len(S2)
    
    # Ascending order
    order_asc = list(range(1, N + 1))
    S3 = greedy_build(order_asc)
    S3 = local_search_add(S3)
    S3 = removal_search(S3, max_iters=100)
    S3 = local_search_add(S3)
    if len(S3) > best_size:
        best = S3
        best_size = len(S3)
    
    # Multiple random orderings
    for iteration in range(300):
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
    best = swap2_search(best, rng, max_iters=10000)
    best = local_search_add(best)
    
    # Try more random restarts with swap
    for iteration in range(100):
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
    
    # Final polish: removal + local + swap2
    best = removal_search(best, max_iters=500)
    best = local_search_add(best)
    best = swap2_search(best, rng, max_iters=15000)
    best = local_search_add(best)
    
    # Verify
    if has_triple(best):
        # Fallback: just use numbers > 90
        best = set(range(91, N + 1))
    
    return list(best)
