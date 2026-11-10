"""Conflict-graph MIS approach: degree-based greedy + 2-for-1 swap local search."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0
    N = 180
    rng = random.Random(20240517)

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if a < b and 1 <= a <= N:
                    triples.append((a, b, c))

    # Build conflict_pairs[x] = list of (y, z) such that {x, y, z} is a forbidden triple
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for a, b, c in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    # Build conflict graph: edge between x,y if they share a third element in some triple
    # degree[x] = number of distinct neighbors
    neighbors = {x: set() for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for (y, z) in conflict_pairs[x]:
            if y != x and z != x:
                neighbors[x].add(y)
                neighbors[x].add(z)
    degree = {x: len(neighbors[x]) for x in range(1, N + 1)}

    # Precompute conflict_count for fast use: conf_count[x][S] is expensive,
    # so we use the pair-based check.
    def count_conflicts(x, S):
        count = 0
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                count += 1
        return count

    def has_triple(S):
        for a in S:
            for (b, c) in conflict_pairs[a]:
                if b in S and c in S:
                    return True
        return False

    # Greedy: process elements in given order, add if no conflict
    def greedy_order(order):
        S = set()
        for x in order:
            if count_conflicts(x, S) == 0:
                S.add(x)
        return S

    # Degree-based ordering: fewer conflicts first
    degree_order = sorted(range(1, N + 1), key=lambda x: degree[x])

    # Phase 1: Multiple greedy constructions
    best = set(range(91, N + 1))  # baseline: 90 elements
    best_size = len(best)

    # Degree-based greedy (fewest conflicts first)
    cand = greedy_order(degree_order)
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)

    # Reverse degree-based (most conflicts first - may leave room)
    cand = greedy_order(list(reversed(degree_order)))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)

    # Descending
    cand = greedy_order(list(range(N, 0, -1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)

    # Ascending
    cand = greedy_order(list(range(1, N + 1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)

    # Random orderings
    for _ in range(300):
        if time.time() >= deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = greedy_order(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)

    # Phase 2: Local search - try to add elements
    def try_add_elements(S):
        improved = True
        while improved:
            if time.time() >= deadline:
                return False
            improved = False
            for x in range(1, N + 1):
                if x in S:
                    continue
                if count_conflicts(x, S) == 0:
                    S.add(x)
                    improved = True
                    break
        return True

    try_add_elements(best)
    best_size = len(best)

    # Phase 3: 2-for-1 swap local search
    # Try removing 2 elements {y1, y2} from S and adding 1 element x not in S.
    # Net gain: -1 (bad), but may enable adding more later.
    # Actually, let's do 1-for-1 first, then 2-for-2.
    
    # 1-for-1 swap: remove y, add x (net 0, but changes structure)
    def one_for_one_swap(S):
        for x in range(1, N + 1):
            if x in S:
                continue
            # Count conflicts of x in S
            c_in_S = count_conflicts(x, S)
            if c_in_S == 0:
                continue  # just add directly
            # Try removing one element y to resolve all conflicts
            for y in list(S):
                # Check if removing y resolves all conflicts of x
                resolved = True
                for (u, v) in conflict_pairs[x]:
                    if u in S and v in S:
                        if u != y and v != y:
                            resolved = False
                            break
                if resolved:
                    S.discard(y)
                    S.add(x)
                    return True
        return False

    # 2-for-2 swap: remove {y1,y2}, add {x1,x2}
    def two_for_two_swap(S):
        not_in_S = [x for x in range(1, N + 1) if x not in S]
        if len(not_in_S) < 2:
            return False
        for i in range(len(not_in_S)):
            x1 = not_in_S[i]
            c1 = count_conflicts(x1, S)
            if c1 == 0:
                continue
            for j in range(i + 1, len(not_in_S)):
                x2 = not_in_S[j]
                c2 = count_conflicts(x2, S)
                if c2 == 0:
                    continue
                # Find pairs (y1, y2) in S whose removal resolves all conflicts of x1 and x2
                # This is expensive; use heuristic: collect all conflicting elements
                conflict_elems_x1 = set()
                for (u, v) in conflict_pairs[x1]:
                    if u in S and v in S:
                        conflict_elems_x1.add(u)
                        conflict_elems_x1.add(v)
                conflict_elems_x2 = set()
                for (u, v) in conflict_pairs[x2]:
                    if u in S and v in S:
                        conflict_elems_x2.add(u)
                        conflict_elems_x2.add(v)
                # Need to remove a subset of S that hits all conflict pairs for both x1 and x2
                # Try removing exactly 2 elements
                all_conf = conflict_elems_x1 | conflict_elems_x2
                if len(all_conf) > 4:
                    continue
                # Try all pairs from all_conf (or from S if small)
                candidates = list(all_conf) if len(all_conf) <= 6 else list(S)[:20]
                for p in range(len(candidates)):
                    y1 = candidates[p]
                    for q in range(p + 1, len(candidates)):
                        y2 = candidates[q]
                        # Check if removing y1,y2 resolves all conflicts for x1
                        ok1 = True
                        for (u, v) in conflict_pairs[x1]:
                            if u in S and v in S:
                                if u != y1 and u != y2 and v != y1 and v != y2:
                                    ok1 = False
                                    break
                        if not ok1:
                            continue
                        ok2 = True
                        for (u, v) in conflict_pairs[x2]:
                            if u in S and v in S:
                                if u != y1 and u != y2 and v != y1 and v != y2:
                                    ok2 = False
                                    break
                        if not ok2:
                            continue
                        # Also check x1 and x2 don't conflict with each other after removal
                        # (they don't share a triple element that's in S-{y1,y2}... actually need to check)
                        S.discard(y1)
                        S.discard(y2)
                        S.add(x1)
                        S.add(x2)
                        return True
        return False

    # Iterative local search
    for _ in range(20000):
        if time.time() >= deadline:
            break
        progress = False
        # Try direct additions
        if try_add_elements(best):
            progress = True
        # Try 1-for-1 swaps
        if one_for_one_swap(best):
            progress = True
            try_add_elements(best)
        # Try 2-for-2 swaps
        if two_for_two_swap(best):
            progress = True
            try_add_elements(best)
        if not progress:
            break

    # Phase 4: Random restarts with diverse seeds
    for _ in range(3000):
        if time.time() >= deadline:
            break
        size = rng.randint(50, 110)
        s = set(rng.sample(range(1, N + 1), size))
        if has_triple(s):
            continue
        remaining = [x for x in range(1, N + 1) if x not in s]
        rng.shuffle(remaining)
        for x in remaining:
            if time.time() >= deadline:
                break
            if count_conflicts(x, s) == 0:
                s.add(x)
        # Local polish
        try_add_elements(s)
        if one_for_one_swap(s):
            try_add_elements(s)
        if len(s) > best_size:
            best = s
            best_size = len(s)

    # Final verification
    if has_triple(best):
        best = set(range(91, N + 1))

    return list(best)
# EVOLVE-BLOCK-END
