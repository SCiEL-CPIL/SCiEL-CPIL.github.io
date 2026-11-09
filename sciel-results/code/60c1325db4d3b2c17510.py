"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    DEADLINE = 120.0  # evaluator limit is 120s; keep safety margin
    t0 = time.time()

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_longest_cycle(adj):
        """Find a reasonably long cycle using greedy path extension with random restarts."""
        if not adj:
            return None
        vertices = list(adj.keys())
        if len(vertices) < 3:
            return None

        best_cycle = []
        # Limit restarts to balance quality and time
        max_restarts = 100
        
        for _ in range(max_restarts):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Random start
            start = random.choice(vertices)
            visited = {start}
            path = [start]
            current_node = start
            
            # Greedy extension
            while True:
                # Get unvisited neighbors
                neighbors = [nb for nb in adj[current_node] if nb not in visited]
                
                if not neighbors:
                    break
                
                # Check if we can close the cycle immediately
                if start in neighbors and len(path) >= 3:
                    cycle = path[:]
                    if len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    break
                
                # Heuristic: Prefer neighbors that are not leaves (degree > 1 in current adj)
                # to avoid getting stuck in dead ends
                non_leaves = [nb for nb in neighbors if len(adj[nb]) > 1]
                if non_leaves:
                    next_node = random.choice(non_leaves)
                else:
                    next_node = random.choice(neighbors)
                
                visited.add(next_node)
                path.append(next_node)
                current_node = next_node
            
            # Check if we can close the cycle from the end of the path
            if len(path) >= 3:
                end_node = path[-1]
                if start in adj[end_node]:
                    cycle = path[:]
                    if len(cycle) > len(best_cycle):
                        best_cycle = cycle

        return best_cycle if len(best_cycle) >= 3 else None

    def decompose(edges):
        """Decompose edge set into cycles and single edges, minimizing piece count."""
        if not edges:
            return []
        remaining = set(edges)
        adj = build_adj(edges)
        pieces = []

        while remaining:
            if time.time() - t0 > DEADLINE - 1:
                break

            # Try to find the longest cycle in the remaining graph
            cycle = find_longest_cycle(adj)
            if cycle and len(cycle) >= 3:
                # Verify all edges in the cycle are in remaining
                valid = True
                cycle_edges = []
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e not in remaining:
                        valid = False
                        break
                    cycle_edges.append(e)
                if valid:
                    # Remove cycle edges from remaining
                    for e in cycle_edges:
                        remaining.discard(e)
                        u, v = e
                        adj[u].discard(v)
                        adj[v].discard(u)
                    pieces.append(cycle)
                    continue

            # No valid cycle found; take an arbitrary edge as a single-edge piece
            e = next(iter(remaining))
            u, v = e
            remaining.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 1:
            # Fallback: return all edges as single pieces for remaining graphs
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
