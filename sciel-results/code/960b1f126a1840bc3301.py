import time

_DEADLINE = 140.0
_START = time.time()

def _remaining():
    return _DEADLINE - time.time()

def construct():
    from random import Random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        rng = Random(seed)
        # Build cross adjacency: for each C-vertex u, the set of I-vertices it connects to
        cross = [set() for _ in range(k)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross[u].add(v)

        # Cross-edges remaining to cover
        cross_edges = set()
        for u in range(k):
            for v in cross[u]:
                cross_edges.add((u, v))

        # Star sizes (how many I-neighbors each C-vertex has)
        star_size = [len(cross[u]) for u in range(k)]
        order = sorted(range(k), key=lambda u: (-star_size[u], u))

        cliques = []
        used = [False] * n  # whether a vertex is "consumed" by a mixed clique

        # Greedily pack mixed cliques {u} ∪ S where S ⊆ cross[u], disjoint from prior
        for u in order:
            if _remaining() < 1.0:
                break
            if not cross[u]:
                continue
            S = sorted(cross[u])
            cliques.append([u] + S)
            for v in S:
                used[v] = True
            # remove all edges (u, v) for v in S from cross_edges
            for v in S:
                cross_edges.discard((u, v))

        # Now cross_edges only contains edges (u, v) where u's mixed clique didn't include v
        # (which can't happen since we took all of cross[u]), OR edges from other C-vertices
        # Wait: each cross edge (u,v) is only adjacent to u on the C-side. When we made
        # clique {u}+cross[u], we covered ALL edges (u, v) for v in cross[u].
        # So cross_edges should now be empty. Let's verify and handle residual.
        if cross_edges:
            # Residual: cover each remaining edge as a 2-clique
            for (u, v) in sorted(cross_edges):
                if _remaining() < 0.5:
                    break
                cliques.append([u, v])

        # Cover C-internal edges: one clique on all of C (it's a clique since all C-edges exist)
        if k >= 2:
            cliques.append(list(range(k)))

        # Validate: every listed set is a clique, every edge in exactly one clique
        # Build edge -> count map
        edge_count = defaultdict(int)
        for clique in cliques:
            cl = sorted(clique)
            if len(cl) < 2:
                # invalid, but shouldn't happen
                continue
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    edge_count[(cl[i], cl[j])] += 1

        # Check all original edges are covered exactly once
        # Rebuild original edge set
        rng2 = Random(seed)
        orig_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                orig_edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng2.random() < p:
                    orig_edges.add((u, v))

        # Verify
        for e in orig_edges:
            if edge_count[e] != 1:
                # Fallback: brute force 2-clique cover
                return [[list(e) for e in orig_edges]]
        for e in edge_count:
            if e not in orig_edges:
                # Fallback
                return [[list(e) for e in orig_edges]]

        # Convert to pure Python int lists
        result = []
        for clique in cliques:
            result.append([int(x) for x in clique])
        return result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if _remaining() < 5.0:
            # Emergency fallback: 2-clique cover
            n, k, p, seed = spec
            rng = Random(seed)
            edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
            edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
            results.append([[int(u), int(v)] for (u, v) in sorted(edges)])
            continue
        results.append(solve(*spec))

    return results
