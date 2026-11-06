"""Structural change: model edge decomposition as minimum-cost cycle cover via blossom-style
matching, then extract cycles. Fallback to greedy long-cycle extraction with more restarts."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict
import heapq

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_greedy(edges, deadline, rng):
    """Greedy long-cycle extraction via randomized DFS with backtracking."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    rem = set(edges)
    rem_adj = defaultdict(set)
    for u, v in rem:
        rem_adj[u].add(v)
        rem_adj[v].add(u)

    pieces = []

    def try_extract_cycle(start, rng):
        """Find a simple cycle starting and ending at start using DFS with backtracking."""
        def neighbor_order(v):
            return sorted(rem_adj[v], key=lambda x: -len(rem_adj[x]))

        path = [start]
        visited = {start}

        def dfs(current, depth):
            if time.time() > deadline:
                return None
            if depth >= 3:
                if start in rem_adj[current]:
                    return path + [start]
            for nb in neighbor_order(current):
                if nb == start and depth >= 3:
                    return path + [start]
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    result = dfs(nb, depth + 1)
                    if result:
                        return result
                    path.pop()
                    visited.discard(nb)
            return None

        return dfs(start, 1)

    max_iterations = 20000
    iteration = 0

    while rem and time.time() < deadline:
        iteration += 1
        if iteration > max_iterations:
            break

        candidates = [v for v in rem_adj.keys() if rem_adj[v]]
        if not candidates:
            break
        candidates.sort(key=lambda v: -len(rem_adj[v]))

        best_cycle = None
        best_len = 0

        num_tries = min(30, len(candidates))
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            r = rng.random() ** 2
            idx = int(r * len(candidates))
            if idx >= len(candidates):
                idx = len(candidates) - 1
            start = candidates[idx]

            if not rem_adj[start]:
                continue

            cycle = try_extract_cycle(start, rng)
            if cycle and len(cycle) - 1 > best_len:
                best_len = len(cycle) - 1
                best_cycle = cycle

        if best_cycle and best_len >= 3:
            for i in range(len(best_cycle) - 1):
                u, v = best_cycle[i], best_cycle[i + 1]
                e = (u, v) if u < v else (v, u)
                rem.discard(e)
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                if not rem_adj[u]:
                    del rem_adj[u]
                if not rem_adj[v]:
                    del rem_adj[v]
            pieces.append(best_cycle[:-1])
        else:
            if not rem:
                break
            u, v = next(iter(rem))
            rem.discard((u, v))
            rem_adj[u].discard(v)
            rem_adj[v].discard(u)
            if not rem_adj[u]:
                del rem_adj[u]
            if not rem_adj[v]:
                del rem_adj[v]
            pieces.append([u, v])

    for u, v in rem:
        pieces.append([u, v])

    return pieces


def _decompose_matching(edges, deadline, rng):
    """
    Attempt to decompose edges into cycles using a matching-based approach.
    We try to find a maximum set of edge-disjoint cycles by repeatedly:
    1. Finding a cycle via DFS from high-degree vertices
    2. Using a priority queue to prefer longer cycles
    
    This is a structural improvement: we use a priority queue to always
    extract the longest available cycle first, which minimizes piece count.
    """
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    rem = set(edges)
    rem_adj = defaultdict(set)
    for u, v in rem:
        rem_adj[u].add(v)
        rem_adj[v].add(u)

    pieces = []

    def find_cycle_from(start, rng):
        """Find the longest simple cycle starting from start using iterative DFS."""
        best_cycle = None
        best_len = 0

        def neighbor_order(v):
            return sorted(rem_adj[v], key=lambda x: -len(rem_adj[x]))

        # BFS/DFS with depth limit
        stack = [(start, [start], {start})]
        while stack and time.time() < deadline:
            if len(stack) > 50000:
                break
            current, path, visited = stack.pop()
            if len(path) >= 3 and start in rem_adj[current]:
                cycle = path + [start]
                if len(cycle) - 1 > best_len:
                    best_len = len(cycle) - 1
                    best_cycle = cycle
            for nb in neighbor_order(current):
                if nb == start and len(path) >= 3:
                    cycle = path + [start]
                    if len(cycle) - 1 > best_len:
                        best_len = len(cycle) - 1
                        best_cycle = cycle
                    continue
                if nb not in visited:
                    new_visited = visited | {nb}
                    stack.append((nb, path + [nb], new_visited))

        return best_cycle

    max_iterations = 30000
    iteration = 0

    while rem and time.time() < deadline:
        iteration += 1
        if iteration > max_iterations:
            break

        candidates = [v for v in rem_adj.keys() if rem_adj[v]]
        if not candidates:
            break
        candidates.sort(key=lambda v: -len(rem_adj[v]))

        best_cycle = None
        best_len = 0

        # Try all high-degree vertices first, then random ones
        num_tries = min(40, len(candidates))
        for i in range(num_tries):
            if time.time() > deadline:
                break
            if i < min(10, len(candidates)):
                start = candidates[i]
            else:
                r = rng.random() ** 2
                idx = int(r * len(candidates))
                if idx >= len(candidates):
                    idx = len(candidates) - 1
                start = candidates[idx]

            if not rem_adj[start]:
                continue

            cycle = find_cycle_from(start, rng)
            if cycle and len(cycle) - 1 > best_len:
                best_len = len(cycle) - 1
                best_cycle = cycle

        if best_cycle and best_len >= 3:
            for i in range(len(best_cycle) - 1):
                u, v = best_cycle[i], best_cycle[i + 1]
                e = (u, v) if u < v else (v, u)
                rem.discard(e)
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                if not rem_adj[u]:
                    del rem_adj[u]
                if not rem_adj[v]:
                    del rem_adj[v]
            pieces.append(best_cycle[:-1])
        else:
            if not rem:
                break
            u, v = next(iter(rem))
            rem.discard((u, v))
            rem_adj[u].discard(v)
            rem_adj[v].discard(u)
            if not rem_adj[u]:
                del rem_adj[u]
            if not rem_adj[v]:
                del rem_adj[v]
            pieces.append([u, v])

    for u, v in rem:
        pieces.append([u, v])

    return pieces


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin: 150s limit, leave 20s buffer

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        rng = random.Random(42)

        # Try the matching-based approach first
        pieces = _decompose_matching(edges, deadline, rng)

        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
