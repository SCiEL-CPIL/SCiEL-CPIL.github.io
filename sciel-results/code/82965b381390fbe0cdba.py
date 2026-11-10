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

    def can_add(x, S_pairs):
        """Check if x can be added to S given its current pair set."""
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

    def greedy_build(order):
        """Greedily build a set by adding elements in the given order."""
        S_set = set()
        S_pairs = set()
        for x in order:
            if x not in S_set and can_add(x, S_pairs):
                S_set.add(x)
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
                if x not in S_set and can_add(x, S_pairs):
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
        
        for bp in bad_pairs_for_x[x_in]:
            if bp in S_pairs:
                if x_out not in bp:
                    return S_set, S_pairs, False
        
        S_new = (S_set - {x_out}) | {x_in}
        
        P_new = set()
        for p in S_pairs:
            if x_out not in p:
                P_new.add(p)
        
        for y in S_new:
            if y != x_in:
                P_new.add(frozenset((x_in, y)))
                
        return S_new, P_new, True

    def count_conflicts(x, S_pairs):
        """Count how many bad pairs of x are present in S_pairs."""
        return len(S_pairs.intersection(bad_pairs_for_x[x]))

    def local_search(S_set, S_pairs, max_iter=500):
        """Local search with swaps and additions."""
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
            
            # Try swaps - prioritize elements with more conflicts
            improved = False
            S_list = list(S_set)
            
            # Score each element by how many conflicts it has
            conflict_scores = []
            for x in S_list:
                conflicts = count_conflicts(x, S_pairs)
                conflict_scores.append((conflicts, x))
            
            # Sort by conflicts descending (try removing high-conflict elements first)
            conflict_scores.sort(reverse=True)
            
            # Interleave: first high-conflict, then random
            random.shuffle(S_list)
            ordered_candidates = []
            for score, x in conflict_scores[:min(10, len(S_list))]:
                ordered_candidates.append(x)
            for x in S_list:
                if x not in ordered_candidates:
                    ordered_candidates.append(x)
            
            for x_out in ordered_candidates:
                if time.time() >= deadline:
                    break
                
                # Get candidates sorted by how few conflicts they'd have
                candidates = [x for x in range(1, n + 1) if x not in S_set]
                
                # Score candidates by their potential conflict count after swap
                # Approximate: count conflicts with S_set - {x_out}
                S_temp_pairs = set()
                for p in S_pairs:
                    if x_out not in p:
                        S_temp_pairs.add(p)
                
                candidate_scores = []
                for x_in in candidates:
                    conf = count_conflicts(x_in, S_temp_pairs)
                    candidate_scores.append((conf, x_in))
                
                # Sort by fewest conflicts first
                candidate_scores.sort()
                
                # Try top candidates first, then some random ones
                tried = set()
                for conf, x_in in candidate_scores[:min(20, len(candidates))]:
                    if x_in in tried:
                        continue
                    tried.add(x_in)
                    S_new, P_new, success = try_swap(S_set, S_pairs, x_out, x_in)
                    if success:
                        S_set = S_new
                        S_pairs = P_new
                        improved = True
                        S_set, S_pairs = hill_climb(S_set, S_pairs)
                        if len(S_set) > best_size:
                            best_S = set(S_set)
                            best_P = set(S_pairs)
                            best_size = len(S_set)
                        break
                
                # If no improvement from sorted, try some random candidates
                if not improved:
                    random.shuffle(candidates)
                    for x_in in candidates[:min(10, len(candidates))]:
                        if x_in in tried:
                            continue
                        tried.add(x_in)
                        S_new, P_new, success = try_swap(S_set, S_pairs, x_out, x_in)
                        if success:
                            S_set = S_new
                            S_pairs = P_new
                            improved = True
                            S_set, S_pairs = hill_climb(S_set, S_pairs)
                            if len(S_set) > best_size:
                                best_S = set(S_set)
                                best_P = set(S_pairs)
                                best_size = len(S_set)
                            break
            
            if not improved:
                break
        
        return best_S, best_P

    # Initial heuristic orders
    rng = random.Random(42)
    orders = [
        all_elements,
        list(reversed(all_elements)),
        [x for x in all_elements if x % 2 == 0],
        [x for x in all_elements if x % 2 == 1],
        sorted(all_elements, key=lambda x: -x.bit_count()),
        sorted(all_elements, key=lambda x: x.bit_count()),
        sorted(all_elements, key=lambda x: len([d for d in range(2, x+1) if x % d == 0])),
        rng.sample(all_elements, n),
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
        S_set, S_pairs = local_search(S_set, S_pairs, max_iter=300)
        if len(S_set) > best_size:
            best_S = S_set
            best_P = S_pairs
            best_size = len(S_set)

    # More random restarts with varied kicks
    for _ in range(3000):
        if time.time() >= deadline:
            break
        
        if best_S:
            strategy = rng.random()
            if strategy < 0.25:
                # Small kick: remove 1-3 elements
                kick_size = min(len(best_S), rng.randint(1, 3))
                to_remove = set(rng.sample(list(best_S), kick_size))
            elif strategy < 0.5:
                # Medium kick: remove 4-8 elements
                kick_size = min(len(best_S), rng.randint(4, 8))
                to_remove = set(rng.sample(list(best_S), kick_size))
            elif strategy < 0.7:
                # Large kick: remove 10-20% of set
                kick_size = max(1, min(len(best_S), int(len(best_S) * rng.uniform(0.1, 0.2))))
                to_remove = set(rng.sample(list(best_S), kick_size))
            elif strategy < 0.85:
                # Targeted kick: remove elements with high conflict scores
                S_list = list(best_S)
                conflict_scores = []
                for x in S_list:
                    conflicts = count_conflicts(x, best_P)
                    conflict_scores.append((conflicts, x))
                conflict_scores.sort(reverse=True)
                kick_size = min(len(best_S), rng.randint(3, 8))
                to_remove = set(x for _, x in conflict_scores[:kick_size])
            else:
                # Random restart from scratch
                order = rng.sample(all_elements, n)
                S_set, S_pairs = greedy_build(order)
                S_set, S_pairs = local_search(S_set, S_pairs, max_iter=400)
                if len(S_set) > best_size:
                    best_S = S_set
                    best_P = S_pairs
                    best_size = len(S_set)
                continue
            
            S_set = best_S - to_remove
            S_pairs = make_pairs(S_set)
            
            remaining = deadline - time.time()
            max_iter = max(100, int(remaining * 15))
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

    # Final verification and polish
    if best_S:
        # Verify the set is valid
        S_list = sorted(best_S)
        valid = True
        for i in range(len(S_list)):
            if not valid:
                break
            for j in range(i + 1, len(S_list)):
                if not valid:
                    break
                for k in range(j + 1, len(S_list)):
                    x1, x2, x3 = S_list[i], S_list[j], S_list[k]
                    g1 = gcd_table[x1][x2]
                    g2 = gcd_table[x1][x3]
                    g3 = gcd_table[x2][x3]
                    if g1 == g2 == g3:
                        valid = False
                        break
        
        if valid:
            # Try one more round of hill climbing
            best_S, best_P = hill_climb(best_S, best_P)
        
        return list(best_S)
    else:
        # Fallback: return a small valid set
        return [1, 2, 3]
