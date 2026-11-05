"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    # Deadline guard: finish well before the 150s wall-clock limit.
    t0 = time.time()
    DEADLINE = 135.0  # safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, deadline):
        """
        Decompose a set of undirected edges into cycles and single edges.
        Strategy: repeatedly find long cycles using randomized greedy DFS,
        then leave remaining edges as single-edge pieces.
        """
        # Build adjacency
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))

        pieces = []
        remaining = set(edge_set)

        def find_cycle(adj, remaining, max_depth=200):
            """
            Try to find a cycle by random-walk / DFS from a random start.
            Returns a list of vertices forming a cycle, or None.
            """
            if not remaining:
                return None
            # Pick a random edge as starting point
            start_edge = random.choice(list(remaining))
            start = start_edge[0]
            # We want to find a path from start back to start using only remaining edges
            # Use iterative deepening / greedy walk
            for attempt in range(50):
                path = [start]
                visited = {start}
                current = start
                found = False
                for _ in range(max_depth):
                    # Get neighbors of current via remaining edges
                    neighbors = [v for v in adj[current] if (min(current, v), max(current, v)) in remaining]
                    if not neighbors:
                        break
                    # Prefer unvisited, but allow closing the cycle if we hit start
                    unvisited = [v for v in neighbors if v not in visited]
                    if start in neighbors and len(path) >= 3:
                        # Close the cycle
                        found = True
                        break
                    if unvisited:
                        # Random choice among unvisited
                        current = random.choice(unvisited)
                    else:
                        # All neighbors visited; check if we can close
                        if start in neighbors and len(path) >= 3:
                            found = True
                            break
                        break
                    path.append(current)
                    visited.add(current)
                if found:
                    return path
            return None

        # Main loop: extract cycles
        while remaining:
            if time.time() > deadline:
                break
            # Try to find a cycle
            cycle = None
            for _ in range(100):  # multiple attempts
                if time.time() > deadline:
                    break
                # Randomize start by shuffling a copy of remaining
                cycle = find_cycle(adj, remaining)
                if cycle:
                    break
            if cycle:
                # Convert cycle to edge list and remove from remaining
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (u, v) if u < v else (v, u)
                    if e in remaining:
                        cycle_edges.add(e)
                    else:
                        # Invalid cycle; skip
                        cycle = None
                        break
                if cycle and cycle_edges:
                    pieces.append(cycle)
                    remaining -= cycle_edges
            else:
                # No cycle found; break into single edges
                for e in remaining:
                    pieces.append(list(e))
                break

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > DEADLINE:
            # Fallback: return all edges as single pieces for remaining graphs
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        pieces = decompose(edges, DEADLINE - 5.0)  # leave 5s margin
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
