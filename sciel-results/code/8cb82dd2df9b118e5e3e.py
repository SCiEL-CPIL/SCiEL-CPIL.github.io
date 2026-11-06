import time
from collections import deque
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(n, edges):
        """Find a long cycle in the graph using randomized DFS."""
        if not edges:
            return None
        
        # Build adjacency
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Try multiple random starting points and orderings
        best_cycle = None
        rng = Random(42)
        
        for attempt in range(50):
            # Pick a random vertex with degree >= 2
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                break
            start = rng.choice(candidates)
            
            # DFS to find a cycle
            # State: (current, prev, path, visited_set)
            # We want to find a cycle back to start
            def dfs_cycle(curr, prev, path, visited):
                # If we're back at start and path length >= 3, we found a cycle
                if curr == start and len(path) >= 3:
                    return path
                for nei in adj[curr]:
                    if nei == prev:
                        continue
                    if nei == start:
                        if len(path) >= 3:
                            return path
                    elif nei not in visited:
                        visited.add(nei)
                        path.append(nei)
                        result = dfs_cycle(nei, curr, path, visited)
                        if result:
                            return result
                        path.pop()
                        visited.remove(nei)
                return None
            
            visited = {start}
            path = [start]
            cycle = dfs_cycle(start, -1, path, visited)
            
            if cycle:
                if best_cycle is None or len(cycle) > len(best_cycle):
                    best_cycle = cycle
                    if len(best_cycle) > n * 0.7:
                        break
                        
        return best_cycle

    def decompose(edges, n):
        remaining = set(edges)
        pieces = []
        
        # Greedily extract long cycles
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if not remaining:
                break
            
            # Try to find a cycle
            cycle = find_longest_cycle(n, remaining)
            
            if not cycle:
                break
            
            # Add cycle to pieces
            pieces.append(cycle)
            
            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
        
        # Remaining edges are singletons
        for u, v in sorted(remaining):
            pieces.append([u, v])
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
