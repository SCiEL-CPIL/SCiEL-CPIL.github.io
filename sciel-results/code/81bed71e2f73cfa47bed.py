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


def _hopcroft_karp_cycles(adj, n):
    """
    Use Hopcroft-Karp on the line graph to find maximum-cardinality
    edge-disjoint cycle packing.
    
    For each edge e=(u,v) in the graph, we create a node in a bipartite graph.
    We connect e=(u,v) on the left to f=(w,x) on the right if they share a vertex
    and the combined path is consistent (i.e., u=w or u=x, and the other endpoints
    are distinct).
    
    A matching in this bipartite graph corresponds to a set of edge-disjoint paths
    of length 2. Cycles form when these length-2 paths link up.
    
    Actually, a cleaner approach: 
    - For each vertex v, its incident edges form a "star".
    - We want to select pairs of edges at each vertex (a matching on the line graph 
      restricted to edges sharing a vertex) such that the selected pairs form cycles.
    
    Simpler formulation:
    - Create bipartite graph: left side = edges, right side = edges.
    - Connect left edge e1=(u,v) to right edge e2=(w,x) if:
      * They share exactly one vertex, say v
      * u != w and u != x (so they don't form a trivial 2-edge "cycle")
      Actually: they share vertex v, and the other endpoints are different.
      So e1=(u,v) and e2=(v,w) with u != w.
    - A matching in this bipartite graph selects edge-disjoint "turns" at vertices.
    - The selected turns form paths. A cycle is a closed path.
    
    This is getting complex. Let me use a different approach.
    
    Alternative: Iteratively find the longest cycle using DFS, remove it, repeat.
    This is simpler and often good in practice.
    """
    pass


def _find_longest_cycle_dfs(adj, deg, n, max_depth=30):
    """
    Find a long cycle using DFS with backtracking.
    Returns a cycle as list of vertices, or None.
    """
    candidates = [v for v in range(n) if deg[v] >= 2]
    if not candidates:
        return None
    candidates.sort(key=lambda v: -deg[v])
    
    best_cycle = None
    best_len = 2  # need at least 3
    
    for start in candidates[:5]:  # try top 5 start vertices
        if deg[start] < 2:
            continue
        # DFS with path
        path = [start]
        visited = {start}
        best_for_this_start = None
        
        def dfs(node, path, visited):
            nonlocal best_cycle, best_len, best_for_this_start
            for nb in sorted(adj[node], key=lambda x: -deg[x]):
                if nb == start and len(path) >= 3:
                    cycle = list(path)
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                        best_for_this_start = cycle
                    return
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.discard(nb)
        
        dfs(start, path, visited)
        if best_for_this_start:
            return best_cycle
    
    return best_cycle


def _find_cycle_bfs(adj, deg, n):
    """
    Find a cycle using BFS. Returns cycle as list of vertices, or None.
    """
    all_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                all_edges.append((u, v))
    
    if not all_edges:
        return None
    
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        # BFS from v to find path back to u, avoiding direct edge
        queue = deque([(v, [v])])
        visited = {v}
        while queue:
            node, path = queue.popleft()
            for nb in sorted(adj[node], key=lambda x: -deg[x]):
                if nb == u and len(path) >= 2:
                    return path + [u]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
    return None


def _decompose_graph(edges, n, time_limit):
    """
    Decompose a graph into cycles and edges, minimizing piece count.
    Strategy:
    1. Repeatedly find the longest cycle (DFS-based), remove it.
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
        cycle = _find_longest_cycle_dfs(adj, deg, n)
        if cycle is None:
            cycle = _find_cycle_bfs(adj, deg, n)
        
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
