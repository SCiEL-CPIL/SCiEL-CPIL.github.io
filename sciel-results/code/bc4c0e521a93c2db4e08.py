def construct():
    import time
    import random
    from math import gcd

    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    n = 128
    all_elements = list(range(1, n + 1))

    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            gcd_table[i][j] = gcd(i, j)

    # Precompute bad pairs for each x:
    # bad_pairs_for_x[x] = set of frozenset({a, b}) such that (a, b, x) is a bad triple
    # i.e., gcd(a,b) == gcd(a,x) == gcd(b,x)
    bad_pairs_for_x = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, n + 1):
                g_ac = gcd_table[a][c]
                g_bc = gcd_table[b][c]
                if g_ab == g_ac == g_bc:
                    bad_pairs_for_x[c].add(frozenset((a, b)))
                    bad_pairs_for_x[b].add(frozenset((a, c)))
                    bad_pairs_for_x[a].add(frozenset((b, c)))

    def can_add(x, S_set, S_pairs):
        """Check if x can be added to S (represented as set and set of frozenset pairs)."""
        # Check intersection of existing pairs with bad pairs for x
        if S_pairs.intersection(bad_pairs_for_x[x]):
            return False
        return True

    def make_pairs(S):
        """Create set of frozenset pairs from list S."""
        pairs = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                pairs.add(frozenset((S_list[i], S_list[j])))
        return pairs

    def greedy_build(order, S_set=None, S_pairs=None):
        """Greedily build a set by adding elements in the given order."""
        if S_set is None:
            S_set = set()
        if S_pairs is None:
            S_pairs = set()
        for x in order:
            if x not in S_set and can_add(x, S_set, S_pairs):
                S_set.add(x)
                # Update pairs: add (x, y) for all y in S_set \ {x}
                for y in S_set:
                    if y != x:
                        S_pairs.add(frozenset((x, y)))
        return S_set, S_pairs

    def hill_climb(S_set, S_pairs):
        """Try to add any element that can be added, repeatedly."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if x not in S_set and can_add(x, S_set, S_pairs):
                    S_set.add(x)
                    for y in S_set:
                        if y != x:
                            S_pairs.add(frozenset((x, y)))
                    improved = True
        return S_set, S_pairs

    def try_swap(S_set, S_pairs, x_out, x_in):
        """Try to swap x_out for x_in. Returns (new_S_set, new_S_pairs, success)."""
        if x_in in S_set:
            return S_set, S_pairs, False
        
        # We need to check if x_in can be added to S_temp = S_set - {x_out}
        # The pairs in S_temp are P_temp = {p in S_pairs | x_out not in p}
        # We check if any p in P_temp is in bad_pairs_for_x[x_in]
        
        # Optimization: Instead of building P_temp, iterate over bad_pairs_for_x[x_in]
        # and check if they are in S_pairs AND do not involve x_out.
        # If a bad pair (a,b) is in S_pairs and x_out is not in {a,b}, then it's in P_temp.
        
        for bp in bad_pairs_for_x[x_in]:
            if bp in S_pairs:
                # Check if x_out is part of this pair
                if x_out not in bp:
                    return S_set, S_pairs, False
        
        # If we get here, the swap is valid.
        # Construct new S and P efficiently.
        S_new = (S_set - {x_out}) | {x_in}
        
        # P_new = (S_pairs - {pairs involving x_out}) | {pairs involving x_in}
        P_new = set()
        # Keep pairs not involving x_out
        for p in S_pairs:
            if x_out not in p:
                P_new.add(p)
        
        # Add pairs involving x_in with all other elements in S_new
        for y in S_new:
            if y != x_in:
                P_new.add(frozenset((x_in, y)))
                
        return S_new, P_new, True

    def local_search(S_set, S_pairs, max_iter=500):
        """Local search with swaps, 2-for-2 swaps, and additions."""
        best_S = set(S_set)
        best_P = set(S_pairs)
        best_size = len(S_set)
        
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            
            # Try to add elements
            S_set, S_pairs = hill_climb(S_set, S_pairs)
            if len(S_set) > best_size:
                best_S = set(S_set)
                best_P = set(S_pairs)
                best_size = len(S_set)
            
            # Try 1-for-1 swaps
            improved = False
            S_list = list(S_set)
            random.shuffle(S_list)
            for x_out in S_list:
                if time.time() >= deadline:
                    break
                # Try to find a good x_in
                candidates = [x for x in range(1, n + 1) if x not in S_set]
                random.shuffle(candidates)
                for x_in in candidates:
                    S_new, P_new, success = try_swap(S_set, S_pairs, x_out, x_in)
                    if success:
                        S_set = S_new
                        S_pairs = P_new
                        improved = True
                        # After a successful swap, try to add more
                        S_set, S_pairs = hill_climb(S_set, S_pairs)
                        if len(S_set) > best_size:
                            best_S = set(S_set)
                            best_P = set(S_pairs)
                            best_size = len(S_set)
                        break
            
            # Try 2-for-2 swaps: remove {a,b}, add {c,d}
            if len(S_set) >= 4 and time.time() < deadline:
                S_list2 = list(S_set)
                random.shuffle(S_list2)
                for i in range(min(len(S_list2), 20)):
                    if time.time() >= deadline:
                        break
                    a = S_list2[i]
                    for j in range(i + 1, min(i + 10, len(S_list2))):
                        if time.time() >= deadline:
                            break
                        b = S_list2[j]
                        # Try to add c, d to S - {a, b}
                        S_temp = S_set - {a, b}
                        # Find pairs in S_temp
                        P_temp = set()
                        for p in S_pairs:
                            if a not in p and b not in p:
                                P_temp.add(p)
                        
                        # Find candidates that can be added
                        candidates = [x for x in range(1, n + 1) if x not in S_temp and x not in S_set]
                        random.shuffle(candidates)
                        
                        added = []
                        for x in candidates:
                            if len(added) == 2:
                                break
                            if can_add(x, S_temp | set(added), P_temp):
                                # Check against existing pairs in S_temp
                                conflict = False
                                for p in P_temp:
                                    if p in bad_pairs_for_x[x]:
                                        conflict = True
                                        break
                                if not conflict:
                                    # Check against pairs involving already-added elements
                                    for y in added:
                                        if frozenset((x, y)) in bad_pairs_for_x[x] or frozenset((x, y)) in bad_pairs_for_y if False else False:
                                            conflict = True
                                            break
                                    # More careful check
                                    for y in added:
                                        # Check if (x,y) creates a bad triple with any z in S_temp
                                        for z in S_temp:
                                            if z == x or z == y:
                                                continue
                                            g_xy = gcd_table[x][y]
                                            g_xz = gcd_table[x][z]
                                            g_yz = gcd_table[y][z]
                                            if g_xy == g_xz == g_yz:
                                                conflict = True
                                                break
                                        if conflict:
                                            break
                                if not conflict:
                                    added.append(x)
                        
                        if len(added) == 2:
                            c, d = added
                            # Verify the full set is valid
                            S_new = S_temp | {c, d}
                            # Quick verification
                            valid = True
                            S_list_v = list(S_new)
                            for ii in range(len(S_list_v)):
                                if not valid:
                                    break
                                for jj in range(ii + 1, len(S_list_v)):
                                    if not valid:
                                        break
                                    for kk in range(jj + 1, len(S_list_v)):
                                        x1, x2, x3 = S_list_v[ii], S_list_v[jj], S_list_v[kk]
                                        g1 = gcd_table[x1][x2]
                                        g2 = gcd_table[x1][x3]
                                        g3 = gcd_table[x2][x3]
                                        if g1 == g2 == g3:
                                            valid = False
                                            break
                            if valid:
                                S_set = S_new
                                S_pairs = make_pairs(S_set)
                                improved = True
                                S_set, S_pairs = hill_climb(S_set, S_pairs)
                                if len(S_set) > best_size:
                                    best_S = set(S_set)
                                    best_P = set(S_pairs)
                                    best_size = len(S_set)
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_S, best_P

    # Initial heuristic orders
    rng = random.Random(42)
    # Use simpler, faster heuristics
    orders = [
        all_elements,
        list(reversed(all_elements)),
        [x for x in all_elements if x % 2 == 0],
        [x for x in all_elements if x % 2 == 1],
        sorted(all_elements, key=lambda x: -x.bit_count()),
        sorted(all_elements, key=lambda x: x.bit_count()),
        rng.sample(all_elements, n),
        rng.sample(all_elements, n),
    ]

    best_S = set()
    best_P = set()
    best_size = 0

    for order in orders:
        if time.time() >= deadline:
            break
        S_set, S_pairs = greedy_build(order)
        S_set, S_pairs = local_search(S_set, S_pairs, max_iter=200)
        if len(S_set) > best_size:
            best_S = S_set
            best_P = S_pairs
            best_size = len(S_set)

    # More random restarts with kicks
    for _ in range(2000):
        if time.time() >= deadline:
            break
        
        if best_S:
            # Multiple kick strategies
            strategy = rng.random()
            if strategy < 0.3:
                # Small kick: remove 1-3 elements
                kick_size = min(len(best_S), rng.randint(1, 3))
                to_remove = set(rng.sample(list(best_S), kick_size))
            elif strategy < 0.6:
                # Medium kick: remove 4-8 elements
                kick_size = min(len(best_S), rng.randint(4, 8))
                to_remove = set(rng.sample(list(best_S), kick_size))
            elif strategy < 0.8:
                # Large kick: remove 10-20% of set
                kick_size = max(1, min(len(best_S), int(len(best_S) * rng.uniform(0.1, 0.2))))
                to_remove = set(rng.sample(list(best_S), kick_size))
            else:
                # Random restart from scratch
                order = rng.sample(all_elements, n)
                S_set, S_pairs = greedy_build(order)
                S_set, S_pairs = local_search(S_set, S_pairs, max_iter=300)
                if len(S_set) > best_size:
                    best_S = S_set
                    best_P = S_pairs
                    best_size = len(S_set)
                continue
            
            S_set = best_S - to_remove
            S_pairs = make_pairs(S_set)
            
            # Adaptive max_iter based on remaining time
            remaining = deadline - time.time()
            max_iter = max(50, int(remaining * 10))
            S_set, S_pairs = local_search(S_set, S_pairs, max_iter=max_iter)
            if len(S_set) > best_size:
                best_S = S_set
                best_P = S_pairs
                best_size = len(S_set)
        else:
            order = rng.sample(all_elements, n)
            S_set, S_pairs = greedy_build(order)
            S_set, S_pairs = local_search(S_set, S_pairs, max_iter=500)
            if len(S_set) > best_size:
                best_S = S_set
                best_P = S_pairs
                best_size = len(S_set)

    return list(best_S)
