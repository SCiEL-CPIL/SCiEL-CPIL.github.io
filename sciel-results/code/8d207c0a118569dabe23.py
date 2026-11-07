"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _maximal_bicliques(n, adj, deadline):
    # Generate all maximal bicliques via a Bron-Kerbosch style enumeration.
    # A biclique is (L, R) with L and R disjoint, all cross edges present.
    # Maximal = cannot add any vertex to L or R while preserving the biclique property.
    # We enumerate by choosing an ordered pair of "anchor" vertices (l in L, r in R),
    # then L = {l} U (neighbors of r that are also neighbors of all of L), etc.
    # Simpler robust approach: enumerate all (L, R) by fixing a seed vertex on each side
    # and growing. We use a standard approach: for each ordered pair (a, b) with edge a-b,
    # the maximal biclique containing a on left and b on right is determined by the
    # common-neighbor closure.
    # Define: given seed left set S_L and seed right set S_R (nonempty, disjoint),
    # the maximal biclique is:
    #   L = S_L U {v : v adjacent to all in S_R}
    #   R = S_R U {v : v adjacent to all in S_L}
    # iterated to fixpoint.
    # We enumerate maximal bicliques by considering each edge (a,b) as the "defining"
    # edge and computing the maximal biclique where a is in L and b is in R.
    # Two edges may yield the same biclique; we dedupe.
    bicliques = []
    seen = set()
    for (a, b) in P['edges']:
        # Compute maximal biclique with a in L, b in R.
        # Start: L = {a}, R = {b}
        # Iterate: L = {a} U N(b) U {v: v adj to all R}, R = {b} U N(a) U {v: v adj to all L}
        # Actually the standard: L = {a} U (intersection of N(r) for r in R), R = {b} U (intersection of N(l) for l in L)
        # Let's do fixpoint.
        L = {a}
        R = {b}
        while True:
            newL = {a}
            for r in R:
                newL = newL & set(adj[r])
            newL = newL | {a}
            newR = {b}
            for l in L:
                newR = newR & set(adj[l])
            newR = newR | {b}
            # Ensure disjointness
            newL -= newR
            newR -= newL
            if newL == L and newR == R:
                break
            L = newL
            R = newR
        # Ensure a in L and b in R
        L.add(a)
        R.add(b)
        L -= R
        R -= L
        if not L or not R:
            continue
        # Verify it's a valid biclique
        valid = True
        for l in L:
            for r in R:
                if r not in adj[l]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        # Check maximality: can we add any vertex to L or R?
        # A vertex v can be added to L if v is adjacent to all in R and v not in R.
        # A vertex v can be added to R if v is adjacent to all in L and v not in L.
        # For maximality, neither should be possible.
        # But our fixpoint already ensures this. Let's verify.
        for v in range(n):
            if v in L or v in R:
                continue
            can_add_L = all(v in adj[r] for r in R)
            can_add_R = all(v in adj[l] for l in L)
            if can_add_L or can_add_R:
                # Not maximal; extend
                if can_add_L:
                    L.add(v)
                if can_add_R:
                    R.add(v)
                L -= R
                R -= L
                break
        # Re-verify
        valid = True
        for l in L:
            for r in R:
                if r not in adj[l]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        key = (frozenset(L), frozenset(R))
        if key not in seen:
            seen.add(key)
            bicliques.append((list(L), list(R)))
        if time.time() > deadline:
            break
    return bicliques


def _greedy_cover(remaining_edges, adj, n, deadline):
    # Greedy set cover: repeatedly pick the biclique covering most uncovered edges.
    # First, generate candidate bicliques on the remaining graph.
    # For simplicity, use star bicliques and some larger ones.
    # Actually, let's generate maximal bicliques on the remaining graph.
    # Build adjacency for remaining edges.
    rem_adj = defaultdict(set)
    for (u, v) in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    # Generate maximal bicliques
    candidates = []
    seen = set()
    for (a, b) in remaining_edges:
        L = {a}
        R = {b}
        while True:
            newL = {a}
            for r in R:
                newL = newL & rem_adj[r]
            newL = newL | {a}
            newR = {b}
            for l in L:
                newR = newR & rem_adj[l]
            newR = newR | {b}
            newL -= newR
            newR -= newL
            if newL == L and newR == R:
                break
            L = newL
            R = newR
        L.add(a)
        R.add(b)
        L -= R
        R -= L
        if not L or not R:
            continue
        # Verify
        valid = True
        for l in L:
            for r in R:
                if r not in rem_adj[l]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        key = (frozenset(L), frozenset(R))
        if key not in seen:
            seen.add(key)
            # Compute edges covered
            edges_covered = set()
            for l in L:
                for r in R:
                    if r in rem_adj[l]:
                        edges_covered.add((min(l, r), max(l, r)))
            candidates.append((list(L), list(R), edges_covered))
        if time.time() > deadline:
            break
    # Greedy selection
    covered = set()
    result = []
    rem_set = set(remaining_edges)
    while rem_set:
        best = None
        best_count = -1
        for (L, R, edges_covered) in candidates:
            new_cover = len(edges_covered & rem_set)
            if new_cover > best_count:
                best_count = new_cover
                best = (L, R, edges_covered)
        if best is None or best_count <= 0:
            # Fallback: cover one edge at a time
            e = next(iter(rem_set))
            u, v = e
            result.append(([u], [v]))
            rem_set.discard(e)
            continue
        L, R, edges_covered = best
        result.append((L, R))
        covered |= edges_covered
        rem_set -= edges_covered
        if time.time() > deadline:
            break
    return result


def _branch_and_bound(remaining_edges, adj, n, deadline, best_count):
    # Branch and bound for set cover.
    # This is a simplified version; for large instances, we fall back to greedy.
    # If remaining edges is small, try exact.
    if len(remaining_edges) > 30:
        return None  # Too large for B&B, return None to use greedy
    # Generate candidates
    rem_adj = defaultdict(set)
    for (u, v) in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    candidates = []
    seen = set()
    for (a, b) in remaining_edges:
        L = {a}
        R = {b}
        while True:
            newL = {a}
            for r in R:
                newL = newL & rem_adj[r]
            newL = newL | {a}
            newR = {b}
            for l in L:
                newR = newR & rem_adj[l]
            newR = newR | {b}
            newL -= newR
            newR -= newL
            if newL == L and newR == R:
                break
            L = newL
            R = newR
        L.add(a)
        R.add(b)
        L -= R
        R -= L
        if not L or not R:
            continue
        valid = True
        for l in L:
            for r in R:
                if r not in rem_adj[l]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        key = (frozenset(L), frozenset(R))
        if key not in seen:
            seen.add(key)
            edges_covered = set()
            for l in L:
                for r in R:
                    if r in rem_adj[l]:
                        edges_covered.add((min(l, r), max(l, r)))
            candidates.append((L, R, edges_covered))
    # B&B
    best_solution = None
    best_len = best_count

    def bb(rem, current):
        nonlocal best_solution, best_len
        if time.time() > deadline:
            return
        if not rem:
            if len(current) < best_len:
                best_len = len(current)
                best_solution = list(current)
            return
        # Lower bound: each biclique covers at most max_cover edges
        max_cover = 0
        for (L, R, ec) in candidates:
            c = len(ec & rem)
            if c > max_cover:
                max_cover = c
        if max_cover == 0:
            return
        lb = (len(rem) + max_cover - 1) // max_cover
        if len(current) + lb >= best_len:
            return
        # Branch on the first uncovered edge
        e = next(iter(rem))
        # Find candidates covering e
        for (L, R, ec) in candidates:
            if e in ec:
                new_rem = rem - ec
                bb(new_rem, current + [(L, R)])
        # Also try covering e with a single-edge biclique
        u, v = e
        bb(rem - {e}, current + [([u], [v])])

    bb(set(remaining_edges), [])
    if best_solution is not None:
        return best_solution
    return None


def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s limit
    n = P['n']
    adj = defaultdict(set)
    for (u, v) in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Seed solution: star bicliques
    seed_bicliques = []
    for u in range(n):
        if adj[u]:
            seed_bicliques.append(([u], sorted(adj[u])))
    seed_count = len(seed_bicliques)

    best_bicliques = seed_bicliques
    best_count = seed_count

    # Try to improve with B&B / greedy on the full edge set
    # First, try B&B if the graph is small enough
    all_edges = [(min(u, v), max(u, v)) for (u, v) in P['edges']]
    if len(all_edges) <= 50 and time.time() < deadline:
        result = _branch_and_bound(all_edges, adj, n, deadline, best_count)
        if result is not None:
            best_bicliques = result
            best_count = len(result)

    # If B&B didn't help or wasn't run, try greedy
    if time.time() < deadline:
        # Try greedy on remaining edges not covered by best
        # Actually, let's just try a full greedy from scratch
        greedy_result = _greedy_cover(all_edges, adj, n, deadline)
        if len(greedy_result) < best_count:
            best_bicliques = greedy_result
            best_count = len(greedy_result)

    # Ensure types are correct
    final = []
    for (L, R) in best_bicliques:
        final.append(([int(x) for x in L], [int(x) for x in R]))

    return final
# EVOLVE-BLOCK-END
