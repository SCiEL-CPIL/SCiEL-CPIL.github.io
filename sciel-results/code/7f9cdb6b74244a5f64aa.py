import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle(adj, deg, n, remaining_edges, time_limit=0.05):
        """
        Find a cycle in the graph using randomized greedy DFS.
        Returns list of vertices forming the cycle, or None.
        """
        if not remaining_edges:
            return None

        candidates = [v for v in range(n) if deg[v] >= 2]
        if not candidates:
            return None

        rng = random.Random(int(time.time() * 1000) % 100000)
        best_cycle = None
        best_len = 0

        # Try a few random starts, prioritizing high-degree vertices
        # Sort by degree descending, then take top few
        candidates.sort(key=lambda v: -deg[v])
        # Take top 10 or all if fewer
        top_candidates = candidates[:min(10, len(candidates))]

        start_t = time.time()
        for start in top_candidates:
            if time.time() - start_t > time_limit:
                break

            # Greedy DFS: always pick unvisited neighbor with highest degree
            visited = {start}
            path = [start]
            current = start

            while True:
                # Find unvisited neighbors
                unvisited_neighbors = [v for v in adj[current] if v not in visited and (min(current, v), max(current, v)) in remaining_edges]

                # Check if we can close the cycle
                if start in adj[current] and (min(start, current), max(start, current)) in remaining_edges and len(path) >= 3:
                    cycle = path[:]
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    # Don't break immediately, might find longer? 
                    # Actually, greedy doesn't backtrack, so this is the only cycle from this start.

                if not unvisited_neighbors:
                    break

                # Pick the one with highest degree
                unvisited_neighbors.sort(key=lambda v: -deg[v])
                next_node = unvisited_neighbors[0]
                current = next_node
                visited.add(current)
                path.append(current)

                # Check for cycle closure again after moving
                if start in adj[current] and (min(start, current), max(start, current)) in remaining_edges and len(path) >= 3:
                    cycle = path[:]
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    break

        return best_cycle

    def decompose_graph(edges, n, time_limit):
        """
        Decompose graph into cycles and single edges.
        Strategy: Repeatedly find a cycle, remove it.
        Use a heuristic to pick "good" cycles (longer is better).
        """
        if not edges:
            return []

        adj = defaultdict(set)
        deg = [0] * n
        remaining = set(edges)

        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        pieces = []
        start_t = time.time()

        while remaining:
            if time.time() - start_t > time_limit:
                # Fallback: add remaining edges as singles
                for e in remaining:
                    pieces.append([e[0], e[1]])
                break

            # Find a cycle
            cycle = find_cycle(adj, deg, n, remaining, time_limit=0.05)

            if cycle:
                pieces.append(cycle)
                # Remove edges of the cycle
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        remaining.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
                        deg[u] -= 1
                        deg[v] -= 1
            else:
                # No cycle found, take an edge
                # Pick an edge incident to a high-degree vertex to keep structure
                best_edge = None
                best_deg_sum = -1
                for e in remaining:
                    u, v = e
                    ds = deg[u] + deg[v]
                    if ds > best_deg_sum:
                        best_deg_sum = ds
                        best_edge = e

                if best_edge is None:
                    break

                u, v = best_edge
                pieces.append([u, v])
                remaining.discard(best_edge)
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1

        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    # Allocate time roughly proportional to complexity (n^2 or n^3?)
    # Let's just split time evenly for simplicity, or slightly favor larger graphs
    # n=20, 25, 30. Total time ~115s.
    # Let's give 30s, 40s, 45s? Or just 115/3 ~ 38s each.
    # The cycle finding is local, so it should be fast.
    # The bottleneck is the number of iterations (number of pieces).
    
    time_per_graph = (deadline - time.time()) / 3.0

    for i, spec in enumerate(specs):
        # Adjust time for remaining graphs
        remaining_time = deadline - time.time()
        graphs_left = len(specs) - i
        t_limit = remaining_time / graphs_left - 1.0  # 1s buffer
        if t_limit < 1.0:
            t_limit = 0.5

        edges = instance(*spec)
        n = spec[0]
        
        pieces = decompose_graph(edges, n, t_limit)
        results.append(pieces)

    return results
