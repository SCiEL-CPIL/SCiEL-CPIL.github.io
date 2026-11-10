def construct():
    import time
    import random
    import sys
    from math import gcd
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    n = 128
    all_elements = list(range(1, n + 1))

    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            gcd_table[i][j] = gcd(i, j)

    # Find all bad triples: triples (a,b,c) where gcd(a,b)==gcd(a,c)==gcd(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, n + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triples.append((a, b, c))

    # Build conflict structure: for each element, which other elements form a bad triple with it
    # conflict[x] = set of (y,z) pairs such that (x,y,z) is a bad triple
    conflict = [set() for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        conflict[a].add(frozenset((b, c)))
        conflict[b].add(frozenset((a, c)))
        conflict[c].add(frozenset((a, b)))

    # Degree of each vertex (number of bad triples it participates in)
    degree = [0] * (n + 1)
    for (a, b, c) in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1

    # Precompute: for each element x, the list of pairs (a,b) in conflict[x] as tuples
    # for faster iteration
    conflict_pairs = [list(map(tuple, conflict[x])) for x in range(n + 1)]

    # For quick checking: can_add(x, S) where S is a set
    # x can be added if no bad triple (x, y, z) has both y and z in S
    def can_add(x, S_set):
        for a, b in conflict_pairs[x]:
            if a in S_set and b in S_set:
                return False
        return True

    # Greedy construction
    def greedy_build(order, S_set=None):
        if S_set is None:
            S_set = set()
        for x in order:
            if x not in S_set and can_add(x, S_set):
                S_set.add(x)
        return S_set

    # Hill climb: try to add any element that can be added
    def hill_climb(S_set):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if x not in S_set and can_add(x, S_set):
                    S_set.add(x)
                    improved = True
        return S_set

    # Local search with swaps - optimized
    def local_search(S_set, max_iter=200):
        best_S = set(S_set)
        best_size = len(S_set)
        rng = random.Random(42)
        
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            
            # Try to add elements
            S_set = hill_climb(S_set)
            if len(S_set) > best_size:
                best_S = set(S_set)
                best_size = len(S_set)
            
            # Try 1-for-1 swaps
            improved = False
            S_list = list(S_set)
            rng.shuffle(S_list)
            for x_out in S_list:
                if time.time() >= deadline:
                    break
                candidates = [x for x in range(1, n + 1) if x not in S_set]
                rng.shuffle(candidates)
                for x_in in candidates:
                    # Try swap: remove x_out, add x_in
                    S_temp = S_set - {x_out}
                    if can_add(x_in, S_temp):
                        S_set = S_temp | {x_in}
                        improved = True
                        S_set = hill_climb(S_set)
                        if len(S_set) > best_size:
                            best_S = set(S_set)
                            best_size = len(S_set)
                        break
            
            if not improved:
                break
        
        return best_S

    # Initial heuristic orders
    rng = random.Random(42)
    orders = [
        all_elements,
        list(reversed(all_elements)),
        [x for x in all_elements if x % 2 == 0],
        [x for x in all_elements if x % 2 == 1],
        sorted(all_elements, key=lambda x: -x.bit_count()),
        sorted(all_elements, key=lambda x: x.bit_count()),
        sorted(all_elements, key=lambda x: -degree[x]),
        sorted(all_elements, key=lambda x: degree[x]),
        rng.sample(all_elements, n),
        rng.sample(all_elements, n),
    ]

    best_S = set()
    best_size = 0

    for order in orders:
        if time.time() >= deadline:
            break
        S_set = greedy_build(order)
        S_set = local_search(S_set, max_iter=100)
        if len(S_set) > best_size:
            best_S = S_set
            best_size = len(S_set)

    # More extensive local search with kicks
    for _ in range(5000):
        if time.time() >= deadline:
            break
        
        strategy = rng.random()
        if strategy < 0.25:
            kick_size = min(len(best_S), rng.randint(1, 3))
            to_remove = set(rng.sample(list(best_S), kick_size))
        elif strategy < 0.5:
            kick_size = min(len(best_S), rng.randint(4, 8))
            to_remove = set(rng.sample(list(best_S), kick_size))
        elif strategy < 0.75:
            kick_size = max(1, min(len(best_S), int(len(best_S) * rng.uniform(0.1, 0.2))))
            to_remove = set(rng.sample(list(best_S), kick_size))
        else:
            order = rng.sample(all_elements, n)
            S_set = greedy_build(order)
            S_set = local_search(S_set, max_iter=100)
            if len(S_set) > best_size:
                best_S = S_set
                best_size = len(S_set)
            continue
        
        S_set = best_S - to_remove
        S_set = local_search(S_set, max_iter=100)
        if len(S_set) > best_size:
            best_S = S_set
            best_size = len(S_set)

    # Final polish: try to add any remaining elements
    best_S = hill_climb(best_S)

    return list(best_S)
