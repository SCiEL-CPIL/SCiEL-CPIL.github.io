"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    # Precompute for every subset mask whether it is a clique or independent set.
    # n=14 -> 2^14 = 16384 masks.
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    for mask in range(1, 1 << n):
        # take lowest set bit
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        if is_clique[rest]:
            # v must be adjacent to every vertex in rest
            ok = True
            r = rest
            while r:
                lb2 = r & (-r)
                u = lb2.bit_length() - 1
                if u not in adj[v]:
                    ok = False
                    break
                r ^= lb2
            is_clique[mask] = ok
        else:
            is_clique[mask] = False
        if is_indep[rest]:
            # v must be non-adjacent to every vertex in rest
            ok = True
            r = rest
            while r:
                lb2 = r & (-r)
                u = lb2.bit_length() - 1
                if u in adj[v]:
                    ok = False
                    break
                r ^= lb2
            is_indep[mask] = ok
        else:
            is_indep[mask] = False

    valid = [False] * (1 << n)
    for m in range(1, 1 << n):
        valid[m] = is_clique[m] or is_indep[m]

    # Precompute list of valid masks by popcount for ordering
    valid_masks = [m for m in range(1, 1 << n) if valid[m]]
    # Sort by popcount descending (prefer larger parts first)
    valid_masks.sort(key=lambda m: m.bit_count(), reverse=True)

    full_mask = (1 << n) - 1
    best = n + 1
    best_parts = None

    # Greedy initial solution for a good upper bound
    def greedy():
        remaining = full_mask
        parts = []
        while remaining:
            # pick largest valid subset of remaining
            best_m = 0
            # iterate valid masks, check subset of remaining
            for m in valid_masks:
                if m & remaining == m:
                    if m.bit_count() > best_m.bit_count():
                        best_m = m
                        if best_m == remaining:
                            break
            if best_m == 0:
                # should not happen; singletons are always valid
                lb = remaining & (-remaining)
                best_m = lb
            parts.append(best_m)
            remaining ^= best_m
        return parts

    greedy_parts = greedy()
    best = len(greedy_parts)
    best_parts = list(greedy_parts)

    # Branch and bound: assign parts one at a time
    # State: remaining_mask, current_parts_count, current_parts_list
    # We build parts greedily in a smart order: try to pack remaining into
    # as few valid subsets as possible.

    def solve(remaining, count_so_far, parts_list):
        nonlocal best, best_parts
        if time.time() > DEADLINE:
            return
        if count_so_far >= best:
            return
        if remaining == 0:
            best = count_so_far
            best_parts = list(parts_list)
            return
        # Lower bound: at least 1 more part needed
        # Try to find a valid subset of remaining to use as next part
        # Heuristic: try subsets that cover high-degree vertices first
        # Collect valid subsets of remaining
        candidates = []
        for m in valid_masks:
            if m & remaining == m:
                candidates.append(m)
        if not candidates:
            return
        # Sort candidates: prefer larger, and prefer those containing
        # vertices with high remaining-degree
        def score(m):
            pc = m.bit_count()
            # bonus for covering vertices that are hard to place
            s = pc * 100
            r = m
            while r:
                lb = r & (-r)
                v = lb.bit_length() - 1
                # count neighbors in remaining
                s += len(adj[v] & set(range(n)))  # rough
                r ^= lb
            return -s
        candidates.sort(key=score)
        for m in candidates:
            if time.time() > DEADLINE:
                return
            if count_so_far + 1 >= best:
                break
            parts_list.append(m)
            solve(remaining ^ m, count_so_far + 1, parts_list)
            parts_list.pop()
            if best == 2:  # can't do better than 2 (1 would mean whole graph is clique/indep)
                return

    solve(full_mask, 0, [])

    # Convert mask list to vertex lists
    result = []
    for m in best_parts:
        verts = []
        r = m
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            verts.append(v)
            r ^= lb
        result.append(verts)
    return result
# EVOLVE-BLOCK-END
