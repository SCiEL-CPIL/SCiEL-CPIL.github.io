"""Decompose fixed instance graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    """Return three lists of pieces, one per fixed instance graph."""
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin before 150s kill

    # Generate the three instance graphs
    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [instance(*spec) for spec in specs]

    def decompose(edges, n):
        """Decompose edge set into minimum number of cycles and single edges."""
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Greedy: repeatedly find longest cycle in remaining edges
        remaining = set(tuple(sorted(e)) for e in edges)
        pieces = []

        while remaining:
            if time.time() > deadline:
                break

            # Find a cycle in the remaining graph
            cycle = _find_cycle(adj, remaining, n)
            if cycle:
                pieces.append(cycle)
                # Remove cycle edges from remaining
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    remaining.discard(e)
                    # Update adjacency (we'll rebuild if needed, but for now track)
            else:
                # No cycle found; take one edge
                e = next(iter(remaining))
                remaining.discard(e)
                pieces.append([e[0], e[1]])

        # If we timed out, add all remaining edges as single edges
        if remaining:
            for e in remaining:
                pieces.append([e[0], e[1]])

        return pieces

    def _find_cycle(adj, remaining, n):
        """Find a cycle in the subgraph defined by 'remaining' edges."""
        # Build subgraph adjacency from remaining
        sub_adj = defaultdict(list)
        for u, v in remaining:
            sub_adj[u].append(v)
            sub_adj[v].append(u)

        # Try to find a cycle via DFS from each vertex
        # We want the longest cycle for better packing
        best_cycle = None
        best_len = 0

        for start in range(n):
            if start not in sub_adj:
                continue
            # DFS to find a cycle back to start
            cycle = _dfs_cycle(start, start, sub_adj, {start}, [start])
            if cycle and len(cycle) > best_len:
                best_cycle = cycle
                best_len = len(cycle)
                if best_len >= 10:  # Good enough, don't waste time
                    break

        return best_cycle

    def _dfs_cycle(start, current, adj, visited, path):
        """Find a cycle from current back to start."""
        for neighbor in adj.get(current, []):
            if neighbor == start and len(path) >= 3:
                return path[:]
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                result = _dfs_cycle(start, neighbor, adj, visited, path)
                if result:
                    return result
                path.pop()
                visited.discard(neighbor)
        return None

    results = []
    for i, (edges, spec) in enumerate(zip(graphs, specs)):
        n = spec[0]
        pieces = decompose(edges, n)
        results.append(pieces)
        if time.time() > deadline:
            # Fill remaining with trivial decompositions
            for j in range(i + 1, len(graphs)):
                results.append([[list(e)] for e in graphs[j]])
            break

    return results

# EVOLVE-BLOCK-END
