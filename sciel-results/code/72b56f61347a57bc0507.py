"""Clique partition of split graphs using matching-based pair packing."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        I = list(range(k, n))
        C = list(range(k))

        # Build cross-neighborhoods
        cross = {i: [] for i in I}
        for (a, b) in edges:
            if a < k and b >= k:
                cross[b].append(a)
            elif b < k and a >= k:
                cross[a].append(b)

        # All C-edges as frozensets
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u + 1, k))
        used_C_edges = set()

        mixed_cliques = []

        # Process I-vertices in ascending degree (smallest neighborhoods first)
        # This preserves C-edge density for later large-neighborhood vertices
        for i in sorted(I, key=lambda x: len(cross[x])):
            neighbors = cross[i]
            if not neighbors:
                continue

            matched = set()
            # Build a list of (c, degree_in_remaining) for partner selection
            # We want to pair c_a with c_b where the edge (c_a,c_b) is "cheap"
            # i.e., c_a and c_b have few other uncovered edges

            # Sort neighbors by their remaining degree (ascending) to pair
            # low-degree vertices first — their edges are less "expensive" to consume
            def c_degree(c):
                """Number of uncovered C-edges incident to c."""
                cnt = 0
                for v in range(k):
                    if v != c and frozenset((c, v)) in all_C_edges and frozenset((c, v)) not in used_C_edges:
                        cnt += 1
                return cnt

            # Precompute degrees once for this I-vertex
            degs = {c: c_degree(c) for c in neighbors}
            sorted_neighbors = sorted(neighbors, key=lambda c: degs[c])

            for idx_a in range(len(sorted_neighbors)):
                if sorted_neighbors[idx_a] in matched:
                    continue
                c_a = sorted_neighbors[idx_a]
                # Find best partner: minimize combined degree of the pair
                best_partner = None
                best_cost = float('inf')
                for idx_b in range(idx_a + 1, len(sorted_neighbors)):
                    c_b = sorted_neighbors[idx_b]
                    if c_b in matched:
                        continue
                    e = frozenset((c_a, c_b))
                    if e not in used_C_edges:
                        cost = degs[c_a] + degs[c_b]
                        if cost < best_cost:
                            best_cost = cost
                            best_partner = c_b
                if best_partner is not None:
                    matched.add(c_a)
                    matched.add(best_partner)
                    used_C_edges.add(frozenset((c_a, best_partner)))
                    mixed_cliques.append([i, c_a, best_partner])

            # Singles: unmatched C-neighbors
            for c in neighbors:
                if c not in matched:
                    mixed_cliques.append([i, c])

        # Cover remaining C-edges with pure-C cliques
        remaining_C_edges = all_C_edges - used_C_edges
        pure_cliques = []
        remaining = set(remaining_C_edges)

        while remaining:
            if time.time() > deadline:
                break
            # Find a greedy maximal clique in the graph of remaining edges
            # Start from the vertex with highest degree in remaining
            best_start = None
            best_deg = -1
            for e in remaining:
                for v in e:
                    pass
            # Collect all vertices appearing in remaining edges
            verts_in_rem = set()
            for e in remaining:
                verts_in_rem.update(e)

            for v in verts_in_rem:
                deg = sum(1 for e in remaining if v in e)
                if deg > best_deg:
                    best_deg = deg
                    best_start = v

            if best_start is None:
                break

            # Greedy maximal clique starting from best_start
            clique = {best_start}
            candidates = set(verts_in_rem - {best_start})

            # A vertex v can join clique if all edges (v, u) for u in clique are in remaining
            while candidates:
                # Pick candidate with max edges to current clique in remaining
                best_cand = None
                best_score = -1
                for v in candidates:
                    score = sum(1 for u in clique if frozenset((v, u)) in remaining)
                    if score == len(clique) and score > best_score:
                        best_score = score
                        best_cand = v
                    elif score > best_score:
                        best_score = score
                        best_cand = v
                if best_cand is None or best_score < len(clique):
                    # No candidate connects to all clique members; try to add any
                    # that connects to at least some, or break
                    added = False
                    for v in candidates:
                        score = sum(1 for u in clique if frozenset((v, u)) in remaining)
                        if score == len(clique):
                            clique.add(v)
                            candidates.discard(v)
                            added = True
                            break
                    if not added:
                        break
                else:
                    clique.add(best_cand)
                    candidates.discard(best_cand)

            # Remove all remaining edges within the clique
            clique_set = clique
            for e in list(remaining):
                if e.issubset(clique_set):
                    remaining.discard(e)

            if len(clique) >= 2:
                pure_cliques.append(sorted(clique))

        result = mixed_cliques + pure_cliques
        # Ensure all are lists of Python ints
        return [list(c) for c in result]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
# EVOLVE-BLOCK-END
