"""Clique partition of two fixed dense random graphs."""
import time

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 120.0  # hard kill is 150s; keep safety margin

    def elapsed():
        return time.time() - t_start

    def gen_edges(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def partition(edges, n):
        adj = build_adj(edges, n)
        edge_set = set(edges)
        remaining = set(edge_set)
        cliques = []

        # Precompute static degree for tie-breaking
        deg = [len(adj[i]) for i in range(n)]

        while remaining:
            if elapsed() > DEADLINE - 5.0:
                break

            # Pick a vertex with max degree within remaining-edge subgraph
            # (i.e., count remaining edges incident to it)
            # To be efficient, iterate over remaining edges to find a good seed.
            # Choose the vertex that appears in the most remaining edges.
            best_v = None
            best_cnt = -1
            # Build incident count from remaining edges (sample if huge)
            # remaining is a set of edges; n is small enough.
            inc = {}
            for (u, v) in remaining:
                inc[u] = inc.get(u, 0) + 1
                inc[v] = inc.get(v, 0) + 1
            # pick vertex with max incident count, tie-break by static degree
            for v, c in inc.items():
                key = (c, deg[v])
                if best_v is None or key > (best_cnt, 0):
                    best_v = v
                    best_cnt = c

            if best_v is None:
                break

            # Grow a maximal clique containing best_v using remaining edges
            # Candidate set: neighbors of best_v that are connected by remaining edges
            cand = [v for v in adj[best_v] if (best_v, v) in remaining or (v, best_v) in remaining]

            clique = [best_v]
            # Greedy: repeatedly add the candidate that is connected to ALL current clique members
            # via remaining edges, and has the highest potential to grow the clique.
            while cand:
                if elapsed() > DEADLINE - 5.0:
                    break
                
                # Filter candidates: must be connected to ALL vertices in current clique via remaining edges
                valid_cands = []
                for v in cand:
                    is_valid = True
                    for w in clique:
                        if not ((v, w) in remaining or (w, v) in remaining):
                            is_valid = False
                            break
                    if is_valid:
                        valid_cands.append(v)
                
                if not valid_cands:
                    break

                # Among valid candidates, pick the one that connects to the most other valid candidates
                # This helps form larger cliques.
                best_cand = None
                best_score = -1
                for v in valid_cands:
                    # Count connections to other valid candidates
                    s = 0
                    for u in valid_cands:
                        if u == v:
                            continue
                        if (v, u) in remaining or (u, v) in remaining:
                            s += 1
                    # Tie-break by static degree
                    if best_cand is None or s > best_score or (s == best_score and deg[v] > deg[best_cand]):
                        best_score = s
                        best_cand = v

                clique.append(best_cand)
                cand.remove(best_cand)

            if len(clique) < 2:
                # Fallback: just take one remaining edge
                e = next(iter(remaining))
                clique = [e[0], e[1]]

            # Remove all edges inside this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if (a, b) in remaining:
                        remaining.discard((a, b))
                    elif (b, a) in remaining:
                        remaining.discard((b, a))

            cliques.append(clique)

        # Safety: if deadline hit and edges remain, cover them individually
        if remaining and elapsed() > DEADLINE - 5.0:
            for e in remaining:
                cliques.append(list(e))

        return cliques

    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = gen_edges(n, p, seed)
        cliques = partition(edges, n)
        results.append(cliques)
    return results
# EVOLVE-BLOCK-END
