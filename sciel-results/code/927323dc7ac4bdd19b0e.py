"""Clique partition of split graphs via greedy set-packing with local search."""
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []

        # Build structure
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))

        N = {i: set() for i in range(k, n)}
        cross_edges = set()
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
                cross_edges.add((u, v))
            elif v < k and u >= k:
                N[u].add(v)
                cross_edges.add((v, u))

        # Only consider I-vertices that have at least one cross edge
        active_i = [i for i in range(k, n) if N[i]]

        def build_partition(order, greedy_mode="max"):
            """Build partition given order of I-vertices.
            Each I-vertex i gets one or more cliques {i} ∪ S where S ⊆ N(i).
            S is chosen to maximize coverage of uncovered C-edges.
            """
            covered_C = set()
            cliques = []

            for i in order:
                if not N[i]:
                    continue

                remaining = set(N[i])
                while remaining:
                    # Greedily build S: add vertices that maximize new C-edge coverage
                    S = set()
                    # Try to add vertices in order of how many new C-edges they cover
                    while remaining:
                        best_u = None
                        best_gain = -1
                        for u in remaining:
                            # Count how many C-edges involving u and vertices in S are uncovered
                            gain = 0
                            for v in S:
                                e = (min(u, v), max(u, v))
                                if e not in covered_C and e in C_edges:
                                    gain += 1
                            # Also count C-edges between u and other remaining vertices that could be added
                            # But for simplicity, just use gain relative to S
                            if gain > best_gain:
                                best_gain = gain
                                best_u = u
                        # If no gain, just pick one to make progress
                        if best_gain <= 0:
                            best_u = min(remaining)
                        S.add(best_u)
                        remaining.discard(best_u)
                        # Check if S is too large (all C-edges in S already covered)
                        # We could stop, but let's continue for now

                    if not S:
                        break

                    clique = sorted([i] + list(S))
                    cliques.append(clique)

                    # Mark C-edges within S as covered
                    S_list = sorted(S)
                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            covered_C.add((S_list[a], S_list[b]))

            # Cover remaining C-edges with C-only cliques
            remaining_C = C_edges - covered_C
            if remaining_C:
                rem_adj = defaultdict(set)
                for u, v in remaining_C:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)

                while remaining_C:
                    # Find edge with max common neighbors for largest clique
                    best_edge = None
                    best_common = -1
                    for u, v in remaining_C:
                        if u in rem_adj and v in rem_adj:
                            common = rem_adj[u] & rem_adj[v]
                            if len(common) > best_common:
                                best_common = len(common)
                                best_edge = (u, v)

                    if best_edge is None:
                        u, v = next(iter(remaining_C))
                        S = {u, v}
                    else:
                        u, v = best_edge
                        S = {u, v} | (rem_adj[u] & rem_adj[v])

                    S_list = sorted(S)
                    cliques.append(S_list)

                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            remaining_C.discard((S_list[a], S_list[b]))
                    for node in S_list:
                        rem_adj.pop(node, None)
                        for w in list(rem_adj.keys()):
                            rem_adj[w].discard(node)
                    rem_adj = {u2: v2 for u2, v2 in rem_adj.items() if v2}

            return cliques

        def count_cliques(cliques):
            return len(cliques)

        def validate(cliques, n, k):
            seen_edges = set()
            for clique in cliques:
                if len(clique) < 2:
                    return False
                verts = sorted(clique)
                for a in range(len(verts)):
                    for b in range(a + 1, len(verts)):
                        e = (verts[a], verts[b])
                        if e in seen_edges:
                            return False
                        # Check it's a real edge
                        if e in C_edges:
                            pass
                        elif e in cross_edges:
                            pass
                        else:
                            return False
                        seen_edges.add(e)
            # All C-edges covered
            for e in C_edges:
                if e not in seen_edges:
                    return False
            # All cross edges covered
            for e in cross_edges:
                if e not in seen_edges:
                    return False
            return True

        # Generate multiple random orders and pick best
        best_cliques = None
        best_count = float('inf')

        num_attempts = 30
        time_per_attempt = (DEADLINE - t0 - 5.0) / num_attempts

        for attempt in range(num_attempts):
            if time.time() > DEADLINE - 3.0:
                break

            # Random order of active I-vertices
            order = active_i[:]
            random.Random(seed + attempt * 7919).shuffle(order)

            cliques = build_partition(order)

            if not validate(cliques, n, k):
                continue

            cnt = count_cliques(cliques)
            if cnt < best_count:
                best_count = cnt
                best_cliques = cliques[:]

        # Local search: try to improve by reassigning
        if best_cliques is not None and time.time() < DEADLINE - 10.0:
            for ls_iter in range(200):
                if time.time() > DEADLINE - 5.0:
                    break

                improved = False
                # Parse current structure
                cross_cliques = []  # (i, frozenset(S))
                conly_cliques = []  # frozenset(S)
                for clique in best_cliques:
                    c_verts = frozenset(v for v in clique if v < k)
                    i_verts = [v for v in clique if v >= k]
                    if len(i_verts) == 1:
                        cross_cliques.append((i_verts[0], c_verts))
                    else:
                        conly_cliques.append(c_verts)

                covered_by_cross = set()
                for i, S in cross_cliques:
                    S_list = sorted(S)
                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            covered_by_cross.add((S_list[a], S_list[b]))

                # Try to absorb a C-only clique into a cross clique
                for idx, S_conly in enumerate(conly_cliques):
                    S_conly_list = sorted(S_conly)
                    # Check if all edges in S_conly are not covered by cross
                    all_uncovered = True
                    for a in range(len(S_conly_list)):
                        for b in range(a + 1, len(S_conly_list)):
                            if (S_conly_list[a], S_conly_list[b]) in covered_by_cross:
                                all_uncovered = False
                                break
                        if not all_uncovered:
                            break

                    if not all_uncovered:
                        continue

                    # Try to merge into a cross clique
                    for cidx, (i, S_cross) in enumerate(cross_cliques):
                        S_merged = S_cross | S_conly
                        # Check validity: all C-edges in S_merged must be either
                        # already covered by other cross cliques OR be in S_conly
                        # Actually, we need: for any C-edge in S_merged, if it's not
                        # in S_conly, it must not be covered by another cross clique
                        valid = True
                        for a in range(len(S_merged)):
                            for b in range(a + 1, len(S_merged)):
                                e = (min(list(S_merged)[a], list(S_merged)[b]),
                                     max(list(S_merged)[a], list(S_merged)[b]))
                                # This edge must be covered by exactly one clique
                                # It's covered by this merged clique
                                # Check it's not covered by another cross clique
                                # (except this one, which we're modifying)
                                # This is complex; skip for now
                        # Simplified: just try and validate
                        new_cliques = list(best_cliques)
                        # Remove the C-only clique
                        # Find its index in best_cliques
                        conly_clique_list = sorted(list(S_conly))
                        try:
                            new_cliques.remove(conly_clique_list)
                        except ValueError:
                            continue
                        # Modify the cross clique
                        old_cross = sorted([i] + list(S_cross))
                        try:
                            ci = new_cliques.index(old_cross)
                        except ValueError:
                            continue
                        new_cross = sorted([i] + list(S_merged))
                        new_cliques[ci] = new_cross

                        if validate(new_cliques, n, k):
                            best_cliques = new_cliques
                            improved = True
                            break

                    if improved:
                        break

                if not improved:
                    break

        if best_cliques is None:
            # Fallback: trivial
            return [list(e) for e in edges]

        return best_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        results.append(solve(*spec))

    out = []
    for lst in results:
        cl = []
        for c in lst:
            cl.append([int(x) for x in c])
        out.append(cl)
    return out

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
