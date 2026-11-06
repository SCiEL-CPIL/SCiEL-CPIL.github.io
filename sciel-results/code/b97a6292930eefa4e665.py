"""Decompose random graphs into cycles and edges, minimizing piece count."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges_list, time_budget):
    """Decompose edges into cycles (longer preferred) and single edges."""
    if not edges_list:
        return []

    # Build adjacency
    adj = {}
    edge_set = set()
    for u, v in edges_list:
        edge_set.add((u, v))
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)

    active_edges = set(edge_set)
    pieces = []
    deadline = time.time() + time_budget

    def _find_cycle_randomized(start, n):
        """Find a cycle starting from 'start' using randomized DFS with backtracking.
        Returns (cycle_list, cycle_edge_set) or (None, None)."""
        rng = Random(hash((start, tuple(sorted(active_edges)))) % (2**31))
        best_cycle = None
        best_edges = None

        # Try a few random neighbor orderings
        for attempt in range(8):
            # Shuffle neighbor order
            adj_copy = {k: list(v) for k, v in adj.items()}
            for k in adj_copy:
                rng.shuffle(adj_copy[k])

            # DFS with backtracking
            def dfs(curr, path, used_edges, path_verts, depth):
                nonlocal best_cycle, best_edges
                if depth > 40:
                    return
                if len(path) >= 3:
                    # Check if we can close the cycle
                    close_edge = (min(curr, start), max(curr, start))
                    if close_edge in active_edges and close_edge not in used_edges:
                        cycle = list(path)
                        cycle_edges = set()
                        for i in range(len(cycle)):
                            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                            cycle_edges.add((min(a, b), max(a, b)))
                        if best_cycle is None or len(cycle) > len(best_cycle):
                            best_cycle = cycle
                            best_edges = cycle_edges
                        return

                for nb in adj_copy.get(curr, []):
                    edge_key = (min(curr, nb), max(curr, nb))
                    if edge_key not in active_edges or edge_key in used_edges:
                        continue
                    if nb == start and len(path) < 3:
                        continue
                    if nb in path_verts and nb != start:
                        continue

                    new_path = path + [nb]
                    new_used = used_edges | {edge_key}
                    new_verts = path_verts | {nb}
                    dfs(nb, new_path, new_used, new_verts, depth + 1)

            visited = {start}
            path = [start]
            dfs(start, path, set(), visited, 0)

            if best_cycle and len(best_cycle) > n * 0.6:
                break

        return best_cycle, best_edges

    while active_edges:
        if time.time() > deadline:
            for e in active_edges:
                pieces.append([e[0], e[1]])
            break

        # Find a vertex with degree >= 2 to start cycle search
        start = None
        for e in list(active_edges):
            u, v = e
            if len(adj.get(u, set())) >= 2 or len(adj.get(v, set())) >= 2:
                start = u
                break

        if start is None:
            for e in active_edges:
                pieces.append([e[0], e[1]])
            break

        n = max(adj.keys()) + 1 if adj else 0
        cycle, cycle_edges = _find_cycle_randomized(start, n)

        if cycle and cycle_edges:
            pieces.append(cycle)
            active_edges -= cycle_edges
            for e in cycle_edges:
                u, v = e
                if u in adj and v in adj[u]:
                    adj[u].discard(v)
                if v in adj and u in adj[v]:
                    adj[v].discard(u)
        else:
            # Try other vertices
            found_cycle = False
            for e in list(active_edges):
                u, v = e
                for s in [u, v]:
                    if len(adj.get(s, set())) >= 2:
                        cycle, cycle_edges = _find_cycle_randomized(s, n)
                        if cycle and cycle_edges:
                            pieces.append(cycle)
                            active_edges -= cycle_edges
                            for ce in cycle_edges:
                                a, b = ce
                                if a in adj and b in adj[a]:
                                    adj[a].discard(b)
                                if b in adj and a in adj[b]:
                                    adj[b].discard(a)
                            found_cycle = True
                            break
                if found_cycle:
                    break

            if not found_cycle:
                for e in active_edges:
                    pieces.append([e[0], e[1]])
                break

    return pieces

def construct():
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    total_budget = 140  # Leave 10s margin for 150s limit

    results = []
    per_graph_budget = total_budget / len(specs)

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        if remaining < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        pieces = _decompose(edges, min(per_graph_budget, remaining - 2))
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
