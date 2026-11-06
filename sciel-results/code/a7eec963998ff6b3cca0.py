"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 110.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycles_by_packing(adj, n, deadline):
    """
    Find a large set of edge-disjoint cycles using a matching-based approach.
    
    Idea: At each vertex, we can pair up incident edges. A set of pairs forms
    paths; if the paths form closed loops, they are cycles.
    
    A simpler, effective approach:
    - Repeatedly find a long cycle via DFS with good heuristics.
    - Remove it and repeat.
    
    But to do better, we can use the fact that in a graph with minimum degree >= 2,
    we can find many cycles. Let's use a randomized greedy with restarts.
    """
    pieces = []
    adj_copy = {u: set(adj[u]) for u in range(n)}
    deg = [len(adj_copy[u]) for u in range(n)]
    
    while time.time() < deadline:
        # Check if any edges remain
        remaining = sum(len(adj_copy[u]) for u in range(n)) // 2
        if remaining == 0:
            break
        
        # Try to find a long cycle
        cycle = _find_long_cycle(adj_copy, deg, n, deadline)
        if cycle is not None:
            pieces.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                adj_copy[u].discard(v)
                adj_copy[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        else:
            break
    
    return pieces, adj_copy


def _find_long_cycle(adj, deg, n, deadline):
    """Find a long cycle using randomized DFS with multiple starts."""
    candidates = [v for v in range(n) if deg[v] >= 2]
    if not candidates:
        return None
    
    best_cycle = None
    best_len = 2
    
    # Sort candidates by degree (descending) for better heuristic
    candidates.sort(key=lambda v: -deg[v])
    
    # Try multiple start vertices
    for start in candidates[:10]:
        if time.time() > deadline:
            break
        if deg[start] < 2:
            continue
        
        path = [start]
        visited = {start}
        found = None
        
        def dfs(node, path, visited):
            nonlocal best_cycle, best_len, found
            if time.time() > deadline:
                return
            for nb in sorted(adj[node], key=lambda x: -deg[x]):
                if nb == start and len(path) >= 3:
                    cycle = list(path)
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                        found = cycle
                    return
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.discard(nb)
                    if found:
                        return
        
        dfs(start, path, visited)
        if found:
            return best_cycle
    
    return best_cycle


def _decompose_graph(edges, n, time_limit):
    """
    Decompose a graph into cycles and edges, minimizing piece count.
    Strategy:
    1. Repeatedly find the longest cycle, remove it.
    2. When no more cycles, take remaining edges individually.
    3. Post-process: try to find cycles in the edge-pieces subgraph.
    """
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    pieces = []
    start_time = time.time()
    
    # Main loop: find cycles
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Check if any edges remain
        remaining = sum(len(adj[u]) for u in range(n)) // 2
        if remaining == 0:
            break
        
        # Try to find a long cycle
        cycle = _find_long_cycle(adj, deg, n, time.time() + time_limit * 0.1)
        if cycle is not None:
            # Verify cycle edges exist
            valid = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if v not in adj[u]:
                    valid = False
                    break
            if valid:
                pieces.append(cycle)
                # Remove cycle edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    adj[u].discard(v)
                    adj[v].discard(u)
                    deg[u] -= 1
                    deg[v] -= 1
            else:
                # Cycle invalid, take an edge
                edge_found = None
                for u in range(n):
                    if adj[u]:
                        v = next(iter(adj[u]))
                        edge_found = (u, v)
                        break
                if edge_found is None:
                    break
                u, v = edge_found
                pieces.append([u, v])
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        else:
            # No cycle, take an edge
            edge_found = None
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    edge_found = (u, v)
                    break
            if edge_found is None:
                break
            u, v = edge_found
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1
    
    # Add remaining edges
    for u in range(n):
        for v in adj[u]:
            if u < v:
                pieces.append([u, v])
    
    # Post-process: merge edge pieces into cycles
    pieces = _merge_edge_pieces(pieces, edges, n)
    
    return pieces


def _merge_edge_pieces(pieces, original_edges, n):
    """
    Try to replace some single-edge pieces with cycles.
    """
    edge_piece_set = set()
    cycle_pieces = []
    for p in pieces:
        if len(p) == 2:
            edge_piece_set.add(tuple(sorted(p)))
        else:
            cycle_pieces.append(p)
    
    if len(edge_piece_set) < 3:
        return pieces
    
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edge_piece_set:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    new_cycle_pieces = []
    remaining = set(edge_piece_set)
    improved = True
    
    while improved:
        improved = False
        # Find cycles in remaining edge-piece subgraph
        for u in range(n):
            if deg[u] >= 2:
                queue = deque([(u, [u])])
                visited = {u}
                cycle = None
                while queue and cycle is None:
                    node, path = queue.popleft()
                    for nb in sorted(adj[node], key=lambda x: -deg[x]):
                        if nb == u and len(path) >= 3:
                            cycle = path
                            break
                        if nb not in visited:
                            visited.add(nb)
                            queue.append((nb, path + [nb]))
                if cycle:
                    # Verify
                    cycle_edges = set()
                    ok = True
                    for i in range(len(cycle)):
                        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((a, b)))
                        if e not in remaining:
                            ok = False
                            break
                        cycle_edges.add(e)
                    if ok:
                        new_cycle_pieces.append(cycle)
                        for e in cycle_edges:
                            remaining.discard(e)
                            a, b = e
                            adj[a].discard(b)
                            adj[b].discard(a)
                            deg[a] -= 1
                            deg[b] -= 1
                        improved = True
                        break
    
    # Build final pieces
    final = new_cycle_pieces + cycle_pieces
    for e in remaining:
        final.append(list(e))
    
    return final


def _decompose_with_restarts(edges, n, time_limit, num_restarts=3):
    """
    Try multiple decompositions with different random seeds for tie-breaking.
    """
    best_pieces = None
    best_count = float('inf')
    per_restart = time_limit / max(num_restarts, 1)
    
    for r in range(num_restarts):
        if time.time() > time_limit:
            break
        # Slightly perturb the graph ordering by shuffling edge list
        shuffled_edges = list(edges)
        random.shuffle(shuffled_edges)
        pieces = _decompose_graph(shuffled_edges, n, per_restart)
        count = len(pieces)
        if count < best_count:
            best_count = count
            best_pieces = pieces
    
    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]
    
    return best_pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(n, p, seed)
        graphs_left = len(specs) - i
        graph_time = remaining * 0.85 / max(graphs_left, 1)
        
        num_restarts = 3
        pieces = _decompose_with_restarts(edges, n, graph_time, num_restarts=num_restarts)
        results.append(pieces)
    
    return results
