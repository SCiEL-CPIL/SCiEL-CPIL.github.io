"""
Clique partition via vertex-based greedy best-first edge coverage.
Structural change: instead of edge-coloring on a conflict graph, we directly
construct cliques one at a time, always picking the clique that maximizes
covered uncovered edges. Then local search refines by moving edges.
"""
import time
import random
import sys

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _verify_partition(cliques, edge_list, adj):
    if not cliques:
        return len(edge_list) == 0
    edge_idx = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
    edge_cover = [0] * len(edge_list)
    for clique in cliques:
        if len(clique) < 2:
            return False
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                if not adj[clique[a]][clique[b]]:
                    return False
                e = _edge_key(clique[a], clique[b])
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    for c in edge_cover:
        if c != 1:
            return False
    return True

def _solve_one_graph(adj, edge_list, time_budget=55.0):
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    if not edge_list:
        return []

    # Build adjacency bitmask
    adj_mask = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)

    m = len(edge_list)
    edge_set = set(_edge_key(u, v) for u, v in edge_list)

    def _clique_edges(clique):
        """Return set of edge_keys in this clique."""
        es = set()
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                es.add(_edge_key(clique[i], clique[j]))
        return es

    def _greedy_best_first(uncovered_edges):
        """
        Iteratively pick the clique covering the most uncovered edges.
        uncovered_edges: set of edge_keys not yet covered.
        Returns list of cliques (vertex lists).
        """
        cliques = []
        remaining = set(uncovered_edges)

        while remaining and _remaining() > 5:
            # Try to find a good clique from the remaining edges.
            # Strategy: for each vertex, try to build the largest clique
            # among its neighbors that covers remaining edges.
            # Use a heuristic: pick the vertex with the most remaining incident edges,
            # then greedily build a clique from its neighbors.

            # Find the vertex with the most remaining edges
            vertex_remaining = {}
            for ek in remaining:
                u, v = ek
                vertex_remaining[u] = vertex_remaining.get(u, 0) + 1
                vertex_remaining[v] = vertex_remaining.get(v, 0) + 1

            if not vertex_remaining:
                break

            # Try top-k candidate vertices
            candidates = sorted(vertex_remaining.keys(), key=lambda x: -vertex_remaining[x])[:min(10, len(candidates))]

            best_clique = None
            best_covered = 0

            for seed_v in candidates:
                # Build a clique starting from seed_v
                # Neighbors of seed_v that are also in remaining edges
                neighbors_in_remaining = set()
                for ek in remaining:
                    u, v = ek
                    if u == seed_v:
                        neighbors_in_remaining.add(v)
                    elif v == seed_v:
                        neighbors_in_remaining.add(u)

                if not neighbors_in_remaining:
                    continue

                # Greedily build clique: start with {seed_v}, add neighbors
                # that are adjacent to all current members and have remaining edges
                clique = [seed_v]
                available = set(neighbors_in_remaining)

                # Sort available by degree in remaining for better choices
                def _deg_in_remaining(v):
                    cnt = 0
                    for ek in remaining:
                        if v in ek:
                            cnt += 1
                    return cnt

                # Iteratively add the neighbor with most remaining edges
                # that is adjacent to all current clique members
                while available and _remaining() > 5:
                    # Filter to those adjacent to all clique members
                    valid = set()
                    for v in available:
                        ok = True
                        for c in clique:
                            if not adj[c][v]:
                                ok = False
                                break
                        if ok:
                            valid.add(v)
                    if not valid:
                        break
                    # Pick the one with most remaining edges
                    best_v = max(valid, key=lambda v: _deg_in_remaining(v))
                    clique.append(best_v)
                    available.discard(best_v)

                if len(clique) >= 2:
                    covered = _clique_edges(clique) & remaining
                    if len(covered) > best_covered:
                        best_covered = len(covered)
                        best_clique = clique

            if best_clique is None or best_covered == 0:
                # Fallback: just take a single edge as a 2-clique
                ek = next(iter(remaining))
                best_clique = list(ek)
                best_covered = 1

            cliques.append(sorted(best_clique))
            covered = _clique_edges(best_clique) & remaining
            remaining -= covered

        # Handle any leftover edges as individual 2-cliques
        for ek in remaining:
            cliques.append(sorted(ek))

        return cliques

    # Phase 1: Greedy best-first
    all_edges = set(_edge_key(u, v) for u, v in edge_list)
    cliques = _greedy_best_first(all_edges)

    if not _verify_partition(cliques, edge_list, adj):
        return [[u, v] for u, v in edge_list]

    # Phase 2: Local search - try to merge cliques if union is still a valid clique
    def _local_search(cliques, max_time=30.0):
        start = time.time()
        cliques = [list(c) for c in cliques]
        improved = True
        while improved and time.time() - start < max_time and _remaining() > 10:
            improved = False
            nc = len(cliques)
            for i in range(nc):
                if time.time() - start > max_time:
                    break
                for j in range(i + 1, nc):
                    # Check if union of cliques i and j is still a valid clique
                    union_verts = set(cliques[i]) | set(cliques[j])
                    valid = True
                    for a in range(len(union_verts)):
                        for b in range(a + 1, len(union_verts)):
                            if not adj[list(union_verts)[a]][list(union_verts)[b]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        cliques[i] = sorted(union_verts)
                        cliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return cliques

    remaining_time = time_budget - (time.time() - _START)
    if remaining_time > 10:
        cliques = _local_search(cliques, max_time=min(remaining_time - 5, 35.0))
        if not _verify_partition(cliques, edge_list, adj):
            # Revert to greedy result
            cliques = _greedy_best_first(all_edges)

    # Phase 3: Try to absorb singleton edges into larger cliques
    def _absorb_singletons(cliques, adj, edge_list):
        """Try to merge 2-vertex cliques into larger ones if possible."""
        cliques = [list(c) for c in cliques]
        improved = True
        while improved and _remaining() > 10:
            improved = False
            nc = len(cliques)
            for i in range(nc):
                if _remaining() < 10:
                    break
                if len(cliques[i]) < 2:
                    continue
                for j in range(nc):
                    if i == j:
                        continue
                    if len(cliques[j]) < 2:
                        continue
                    # Try to merge j into i
                    union_verts = set(cliques[i]) | set(cliques[j])
                    valid = True
                    verts_list = list(union_verts)
                    for a in range(len(verts_list)):
                        for b in range(a + 1, len(verts_list)):
                            if not adj[verts_list[a]][verts_list[b]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        cliques[i] = sorted(union_verts)
                        cliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return cliques

    if _remaining() > 15:
        cliques = _absorb_singletons(cliques, adj, edge_list)
        if not _verify_partition(cliques, edge_list, adj):
            # Revert
            cliques = _greedy_best_first(all_edges)

    return cliques

def construct():
    global _START
    _START = time.time()

    results = []

    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue

        adj, edge_list = _generate_graph(n, p, seed)

        time_budget = min(60.0, _remaining() - 15)
        if time_budget < 5:
            time_budget = 5

        cliques = _solve_one_graph(adj, edge_list, time_budget)

        if not _verify_partition(cliques, edge_list, adj):
            cliques = [[u, v] for u, v in edge_list]

        results.append(cliques)

    return results
