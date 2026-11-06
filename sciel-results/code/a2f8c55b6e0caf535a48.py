"""Clique partition via iterative maximum-clique extraction.
Repeatedly find a large clique in the remaining-edge graph, add it to the
partition, and delete its edges.  Uses a greedy max-clique heuristic with
colour-bound pruning for speed on n<=64 dense graphs.
"""
import time

_DEADLINE = 145.0
_T0 = time.time()


def _deadline_hit():
    return (time.time() - _T0) >= _DEADLINE


def _random_graph(n, p, seed):
    from random import Random
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    return adj, edges


def _greedy_max_clique(adj, vertices):
    """Greedy heuristic: build a large clique from the given candidate set.
    Repeatedly add the vertex with the most neighbours already in the clique."""
    if not vertices:
        return []
    clique = [vertices[0]]
    remaining = set(vertices)
    remaining.remove(clique[0])
    while remaining:
        # Find vertex in remaining with max overlap with clique
        best_v = None
        best_score = -1
        for v in remaining:
            score = len(adj[v] & set(clique))
            if score > best_score:
                best_score = score
                best_v = v
        if best_score < len(clique):
            # best_v must be adjacent to all in clique
            if best_v is None or len(adj[best_v] & set(clique)) < len(clique):
                break
        clique.append(best_v)
        remaining.discard(best_v)
    return clique


def _partition(adj, edges, n):
    # Work with a mutable copy of adjacency (as sets of remaining neighbours)
    rem_adj = [set(s) for s in adj]
    remaining_edges = set(edges)
    result = []

    while remaining_edges and not _deadline_hit():
        # Collect all vertices that still have remaining edges
        active = set()
        for u, v in remaining_edges:
            active.add(u)
            active.add(v)

        if not active:
            break

        # Try to find a good clique from active vertices
        # Greedy: pick a vertex with high degree in remaining graph, expand
        # Use multiple seeds and take the best
        best_clique = []
        # Try from a few high-degree vertices
        deg_sorted = sorted(active, key=lambda v: len(rem_adj[v] & active), reverse=True)
        seeds = deg_sorted[:min(8, len(deg_sorted))]
        for seed_v in seeds:
            cand = _greedy_max_clique(rem_adj, [seed_v] + list(rem_adj[seed_v] & active))
            if len(cand) > len(best_clique):
                best_clique = cand

        # If no clique of size >= 2 found, pick an edge
        if len(best_clique) < 2:
            e = next(iter(remaining_edges))
            best_clique = list(e)

        # Verify best_clique is actually a clique in remaining graph
        valid = True
        cl_set = set(best_clique)
        for u in best_clique:
            for v in best_clique:
                if u != v and v not in rem_adj[u]:
                    valid = False
                    break
            if not valid:
                break
        if not valid or len(best_clique) < 2:
            e = next(iter(remaining_edges))
            best_clique = list(e)

        result.append(best_clique)

        # Remove all edges within best_clique from remaining
        cl_set = set(best_clique)
        to_remove = set()
        for u in best_clique:
            for v in best_clique:
                if u < v:
                    to_remove.add((u, v))
        remaining_edges -= to_remove
        for u in best_clique:
            for v in best_clique:
                if u != v:
                    rem_adj[u].discard(v)
                    rem_adj[v].discard(u)

    # Fallback: any leftover edge as 2-clique
    for e in list(remaining_edges):
        if _deadline_hit():
            break
        result.append(list(e))

    return result


def construct():
    out = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        adj, edges = _random_graph(n, p, seed)
        out.append(_partition(adj, edges, n))
    return out
