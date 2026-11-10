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

    # For quick checking: can_add(x, S) where S is a set
    # x can be added if no bad triple (x, y, z) has both y and z in S
    def can_add(x, S_set):
        for pair in conflict[x]:
            a, b = tuple(pair)
            if a in S_set and b in S_set:
                return False
        return True

    # Branch and bound for maximum independent set
    # We use a greedy upper bound: sort remaining vertices by degree, 
    # greedily select vertices, each selected vertex removes its conflicts
    def branch_and_bound(remaining, current_set, current_size, best):
        """
        remaining: set of vertices not yet decided
        current_set: set of vertices included so far
        current_size: len(current_set)
        best: dict with 'size' and 'set' for best found so far
        """
        if not remaining:
            if current_size > best['size']:
                best['size'] = current_size
                best['set'] = set(current_set)
            return

        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best['size']:
            return

        # Greedy upper bound: 
        # Sort remaining by degree (descending)
        # Greedily pick vertices, removing their conflicts from remaining
        rem_list = sorted(remaining, key=lambda x: -degree[x])
        rem_set = set(rem_list)
        greedy_count = 0
        for x in rem_list:
            if x not in rem_set:
                continue
            greedy_count += 1
            # Remove x and all vertices that conflict with x (given current_set)
            rem_set.discard(x)
            for pair in conflict[x]:
                a, b = tuple(pair)
                # If both a and b are in current_set, x is already invalid
                # But we're computing upper bound, so be optimistic
                # Actually, for upper bound, we should only remove vertices that 
                # would definitely conflict. Let's use a simpler bound.
                pass
        
        # Simpler upper bound: current_size + len(remaining)
        # But let's try a better one using the structure
        # A vertex x in remaining can be added if no pair in conflict[x] is both in current_set
        # For upper bound, assume we can add all remaining (optimistic)
        if current_size + len(remaining) <= best['size']:
            return

        # Choose vertex with highest degree in remaining
        x = max(remaining, key=lambda v: degree[v])

        # Branch 1: include x
        new_current = set(current_set)
        new_current.add(x)
        # Remove from remaining all vertices that conflict with x
        # A vertex y conflicts with x if there exists z such that (x,y,z) is bad
        # i.e., if there's a pair (y,z) in conflict[x] with z in new_current
        # Actually, y conflicts with x if adding y to new_current would create a bad triple
        # This happens if there exists z in new_current such that (x,y,z) is bad
        # i.e., if frozenset((x,z)) is in conflict[y] for some z in new_current
        # Or equivalently, if frozenset((y,z)) is in conflict[x] for some z in new_current
        
        # Simpler: y is incompatible with x if there exists z in current_set such that
        # (x,y,z) is a bad triple, i.e., frozenset((x,z)) in conflict[y] or frozenset((y,z)) in conflict[x]
        
        incompatible = set()
        for y in remaining:
            if y == x:
                continue
            # Check if y conflicts with x given current_set
            for pair in conflict[x]:
                a, b = tuple(pair)
                if a == y or b == y:
                    other = b if a == y else a
                    if other in current_set:
                        incompatible.add(y)
                        break
            if y in incompatible:
                continue
            # Also check if x conflicts with y
            for pair in conflict[y]:
                a, b = tuple(pair)
                if a == x or b == x:
                    other = b if a == x else a
                    if other in current_set:
                        incompatible.add(y)
                        break
        
        new_remaining = remaining - {x} - incompatible
        branch_and_bound(new_remaining, new_current, current_size + 1, best)

        # Branch 2: exclude x
        branch_and_bound(remaining - {x}, current_set, current_size, best)

    # Greedy construction
    def greedy_build(order, S_set=None, S_pairs=None):
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

    # Local search with swaps
    def try_swap(S_set, x_out, x_in):
        """Try to swap x_out for x_in. Returns new set if successful, None otherwise."""
        if x_in in S_set:
            return None
        
        S_temp = S_set - {x_out}
        if can_add(x_in, S_temp):
            return S_temp | {x_in}
        return None

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
                    new_S = try_swap(S_set, x_out, x_in)
                    if new_S is not None:
                        S_set = new_S
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

    # Branch and bound from the best found so far
    # Use the complement: try to find a larger set
    if time.time() < deadline - 5:
        # Run branch and bound with a time limit
        # We'll do a simplified version with depth limit
        remaining = set(range(1, n + 1)) - best_S
        # Try to add elements from remaining to best_S using B&B
        # But B&B on 128 vertices is too large, so we do a limited search
        
        # Instead, do more local search with kicks
        for _ in range(3000):
            if time.time() >= deadline:
                break
            
            strategy = rng.random()
            if strategy < 0.3:
                kick_size = min(len(best_S), rng.randint(1, 3))
                to_remove = set(rng.sample(list(best_S), kick_size))
            elif strategy < 0.6:
                kick_size = min(len(best_S), rng.randint(4, 8))
                to_remove = set(rng.sample(list(best_S), kick_size))
            elif strategy < 0.8:
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
