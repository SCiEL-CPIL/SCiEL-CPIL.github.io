"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _remove_edge(adj, remaining, u, v):
    e = (min(u, v), max(u, v))
    remaining.discard(e)
    if v in adj[u]:
        adj[u].remove(v)
    if u in adj[v]:
        adj[v].remove(u)


def _find_cycle_dfs(adj, n, deadline=None):
    """Find a cycle using DFS. Returns list of vertices or None."""
    visited = set()
    
    def dfs(node, path):
        if deadline is not None and time.time() > deadline:
            return None
        for neighbor in adj.get(node, []):
            if neighbor == path[0] and len(path) >= 3:
                return list(path)
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                result = dfs(neighbor, path)
                if result:
                    return result
                path.pop()
                visited.discard(neighbor)
        return None
    
    # Try starting from each vertex with degree >= 2
    candidates = [v for v in adj if len(adj[v]) >= 2]
    random.shuffle(candidates)
    
    for start in candidates[:min(len(candidates), 10)]:
        if deadline is not None and time.time() > deadline:
            break
        visited = {start}
        result = dfs(start, [start])
        if result:
            return result
    return None


def _find_longest_cycle_greedy(adj, n, seed, deadline=None):
    """Find a long cycle using greedy longest-path extension with DFS fallback."""
    if not adj:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    rng.shuffle(vertices)

    # Try multiple starting vertices
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:8] + vertices[:8]))
    rng.shuffle(starts)

    for start in starts[:15]:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # Method 1: Greedy path extension
        path = [start]
        visited = {start}
        
        while True:
            u = path[-1]
            candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
            if not candidates:
                break
            candidates.sort(reverse=True)
            top_k = min(3, len(candidates))
            _, next_v = rng.choice(candidates[:top_k])
            path.append(next_v)
            visited.add(next_v)

        # Check if we can close the cycle directly
        if len(path) >= 3 and start in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)

        # Look for any back-edge to form a cycle
        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(len(path) - 1, i + 1, -1):
                if path[i] in adj[path[j]]:
                    cycle = path[i:j + 1]
                    if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break

    # Method 2: DFS-based cycle finding if greedy didn't find a good one
    if len(best_cycle) < 4:
        dfs_cycle = _find_cycle_dfs(adj, n, deadline)
        if dfs_cycle and len(dfs_cycle) > len(best_cycle):
            best_cycle = dfs_cycle

    return best_cycle


def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []

    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    global_deadline = start_time + time_limit

    def greedy_decompose(remaining_edges, n, seed_offset=0):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        cycle_counter = 0
        while remaining:
            if time.time() > global_deadline:
                # Return what we have plus remaining edges as singles
                for e in remaining:
                    pieces.append(list(e))
                break

            # Find a long cycle
            cycle = _find_longest_cycle_greedy(adj, n, seed + seed_offset + cycle_counter, deadline=global_deadline)
            
            if cycle and len(cycle) >= 3:
                # Verify all edges in cycle exist in remaining
                valid = True
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e not in remaining:
                        valid = False
                        break
                    cycle_edges.add(e)
                
                if valid:
                    # Remove from remaining and adj
                    for e in cycle_edges:
                        u, v = e
                        _remove_edge(adj, remaining, u, v)
                    pieces.append(cycle)
                    cycle_counter += 1
                else:
                    # Cycle invalid, fall through to single edge
                    pass

            if not cycle or len(cycle) < 3:
                # No cycle found; pick a single edge strategically
                # Prefer edges where both endpoints have high degree (removing them
                # may break cycles that couldn't be found)
                # Or pick an edge that's a "bridge" in some sense
                if remaining:
                    # Score edges by sum of endpoint degrees
                    scored = []
                    for e in remaining:
                        u, v = e
                        score = len(adj.get(u, set())) + len(adj.get(v, set()))
                        scored.append((score, e))
                    scored.sort(reverse=True)
                    # Pick from top few randomly for diversity
                    top = scored[:min(5, len(scored))]
                    _, e = random.choice(top)
                    u, v = e
                    _remove_edge(adj, remaining, u, v)
                    pieces.append([u, v])

        return pieces

    # Multi-start search
    for restart in range(300):
        if time.time() > global_deadline:
            break

        pieces = greedy_decompose(edges, n, restart)
        current_count = len(pieces)

        if current_count < best_count:
            best_count = current_count
            best_pieces = pieces

    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_start = time.time()
    total_deadline = total_start + 110.0  # Safety margin
    
    for i, (n, p, seed) in enumerate(specs):
        remaining_time = total_deadline - time.time()
        if remaining_time < 10:
            # Not enough time; use trivial decomposition
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        # Allocate time proportionally, with minimum 10s per graph
        time_limit = min(remaining_time / (3 - i), 45.0)
        time_limit = max(time_limit, 10.0)
        
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=time_limit)
        results.append(pieces)
    
    return results

# EVOLVE-BLOCK-END
