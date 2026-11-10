"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random
from collections import deque, defaultdict

# Deadline guard: the evaluator kills after 150s; we target finishing well before.
_START_TIME = time.time()
_DEADLINE = 125.0  # seconds from start of this module import (safety margin)

def _elapsed():
    return time.time() - _START_TIME

def _time_up():
    return _elapsed() >= _DEADLINE

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        if _time_up():
            # Fallback: return trivial decomposition for remaining graphs
            results.append(_trivial_decompose(spec))
            continue
        edges = instance(*spec)
        n = spec[0]
        pieces = _decompose_graph(n, edges)
        results.append(pieces)

    return results


def _trivial_decompose(spec):
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    return [list(e) for e in instance(*spec)]


def _decompose_graph(n, edges):
    """
    Decompose the edge set into cycles and single edges, minimizing the number of pieces.
    Strategy:
      1. Repeatedly find a cycle in the current graph, remove its edges, add it as a piece.
      2. When no cycle remains (all remaining edges are bridges / acyclic), emit them as single edges.
    To find cycles, we use a heuristic: try to find a long cycle via DFS, or use a 2-core peeling
    approach to focus on cyclic parts.
    """
    # Build adjacency
    adj = defaultdict(set)
    edge_set = set(tuple(sorted(e)) for e in edges)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining_edges = set(edge_set)

    # Work on the graph, repeatedly extracting cycles
    # We maintain a mutable copy of adjacency
    cur_adj = {u: set(v) for u, v in adj.items()}
    for u in range(n):
        if u not in cur_adj:
            cur_adj[u] = set()

    # Remove isolated vertices from consideration
    def get_non_isolated():
        return [u for u in range(n) if cur_adj[u]]

    # Try to find cycles using a modified DFS that looks for back-edges
    # A simpler approach: repeatedly find any cycle via BFS/DFS
    # For small graphs (n<=30), we can afford a few iterations.

    max_iters = 500  # safety
    for _ in range(max_iters):
        if _time_up():
            break
        if not remaining_edges:
            break

        # Check if the remaining graph has any cycle
        # A graph has a cycle iff it has a vertex with degree >= 2 in the 2-core
        # Let's just try to find a cycle via DFS
        cycle = _find_cycle(cur_adj, n)
        if cycle is None:
            break

        # Extract cycle edges
        cycle_edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e in remaining_edges:
                cycle_edges.add(e)
            else:
                # Should not happen if cycle is valid
                break
        else:
            # Valid cycle found
            for e in cycle_edges:
                remaining_edges.discard(e)
                u, v = e
                cur_adj[u].discard(v)
                cur_adj[v].discard(u)
            pieces.append(cycle)
            # Remove isolated vertices
            for u in list(cur_adj.keys()):
                if not cur_adj[u]:
                    del cur_adj[u]
        # If cycle was invalid, break to avoid infinite loop
        if len(cycle_edges) < len(cycle):
            break

    # Emit remaining edges as single-edge pieces
    for e in remaining_edges:
        pieces.append(list(e))

    return pieces


def _find_cycle(adj, n):
    """
    Find a simple cycle in the graph given by adj (dict of sets).
    Returns a list of vertices [v0, v1, ..., vk] where (vk, v0) is an edge,
    or None if no cycle exists.
    Uses DFS with coloring.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {u: WHITE for u in range(n)}
    parent = {u: None for u in range(n)}

    def dfs(start):
        stack = [(start, None, iter(sorted(adj.get(start, []))))]
        color[start] = GRAY
        path = [start]

        while stack:
            node, par, neighbors = stack[-1]
            advanced = False
            for nei in neighbors:
                if nei not in adj.get(node, set()):
                    continue
                if color[nei] == GRAY:
                    # Found a cycle: from nei to node in the path
                    idx = path.index(nei)
                    cycle = path[idx:] + [nei]
                    return cycle
                elif color[nei] == WHITE:
                    color[nei] = GRAY
                    parent[nei] = node
                    path.append(nei)
                    stack.append((nei, node, iter(sorted(adj.get(nei, [])))))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                path.pop()
                stack.pop()
        return None

    for u in range(n):
        if color[u] == WHITE and adj.get(u):
            cycle = dfs(u)
            if cycle is not None:
                return cycle
    return None


# EVOLVE-BLOCK-END
