"""Decompose three fixed random graphs into few cycles and edges (Erdős-Gallai style)."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges, n, deadline):
    """
    Greedy decomposition: repeatedly find cycles in the remaining edge set,
    preferring longer cycles. Remaining edges become single-edge pieces.
    """
    # Build adjacency for quick neighbor lookup
    adj = {v: set() for v in range(n)}
    remaining = set()
    for (u, v) in edges:
        remaining.add((u, v))
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    # We'll keep a set of remaining edges for O(1) lookup
    rem_set = set(remaining)

    def find_cycle(start, min_len=3):
        """
        DFS to find a cycle containing 'start'. Returns list of vertices
        in cycle order (closing edge implied) or None.
        """
        # Try to find a cycle by DFS
        # We want cycles of length >= min_len
        best_cycle = None
        # Use iterative DFS with path tracking
        # For efficiency, limit search depth
        
        # Simple approach: try all neighbors as start of path back to start
        # Actually, let's do a proper DFS looking for a back-edge to start
        
        # Better: do a DFS from start, and when we hit start again with path length >= min_len, we have a cycle
        # But we need to be careful: we're looking for a cycle in the current graph.
        
        # Let's use a different approach: find any cycle in the graph, then check if it's long enough.
        # Actually, for the greedy approach, let's just find ANY cycle and extract it.
        
        # Find a cycle using DFS
        visited = {}  # node -> depth in DFS tree
        parent = {}
        stack = [(start, -1, 0)]
        visited[start] = 0
        parent[start] = -1
        path = [start]
        
        while stack:
            if time.time() > deadline:
                return None
            node, par, depth = stack[-1]
            # Try to find an unvisited neighbor
            found_next = False
            for nb in sorted(adj[node]):  # deterministic
                if nb == par:
                    continue
                if (min(node, nb), max(node, nb)) not in rem_set:
                    continue
                if nb not in visited:
                    visited[nb] = depth + 1
                    parent[nb] = node
                    stack.append((nb, node, depth + 1))
                    path.append(nb)
                    found_next = True
                    break
                else:
                    # Back edge
                    cycle_len = depth - visited[nb] + 1
                    if cycle_len >= min_len:
                        # Extract cycle
                        cycle = [nb]
                        cur = node
                        while cur != nb:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.reverse()
                        return cycle
            if not found_next:
                stack.pop()
                path.pop()
        return None

    # Greedy: repeatedly find the longest cycle we can
    # For efficiency, we'll find cycles and extract them
    # To find longer cycles, we can try starting from different vertices
    
    max_iterations = 200  # safety
    iter_count = 0
    while rem_set and iter_count < max_iterations:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Try to find a cycle
        # Start from vertices with highest degree in remaining graph
        deg = {v: len([nb for nb in adj[v] if (min(v, nb), max(v, nb)) in rem_set]) for v in range(n)}
        
        best_cycle = None
        best_len = 0
        
        # Try a few starting points
        candidates = sorted(range(n), key=lambda v: -deg[v])[:min(10, n)]
        for start in candidates:
            if deg[start] < 2:
                continue
            cycle = find_cycle(start, min_len=3)
            if cycle is not None and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
            if time.time() > deadline:
                break
        
        if best_cycle is None:
            # No more cycles; break
            break
        
        # Extract the cycle's edges from remaining
        for i in range(len(best_cycle)):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % len(best_cycle)]
            e = (min(u, v), max(u, v))
            if e in rem_set:
                rem_set.remove(e)
                adj[u].discard(v)
                adj[v].discard(u)
        pieces.append(best_cycle)
    
    # Remaining edges as single-edge pieces
    for (u, v) in sorted(rem_set):
        pieces.append([u, v])
    
    return pieces

def construct():
    deadline = time.time() + 110  # 110s safety margin within 120s limit
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = _instance(n, p, seed)
        if time.time() > deadline:
            # Fallback: just use all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        pieces = _decompose(edges, n, deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
