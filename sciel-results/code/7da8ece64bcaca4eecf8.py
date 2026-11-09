"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _max_2matching(adj, n):
    """Find maximum 2-matching (max edges in vertex-disjoint cycles) using max-flow.
    
    Reduction: For each vertex v, create v_in and v_out with capacity 2.
    For each edge (u,v), create directed edges u_out->v_in and v_out->u_in with cap 1.
    Max flow gives the 2-matching. Extract cycles from resulting subgraph.
    """
    # Node indexing: v_in = 2*v, v_out = 2*v+1
    # Source = 2*n, Sink = 2*n+1
    src = 2 * n
    sink = 2 * n + 1
    total_nodes = 2 * n + 2
    
    # Build flow network using adjacency list with capacities
    # graph[node] = list of (neighbor, capacity, reverse_index)
    graph = [[] for _ in range(total_nodes)]
    
    def add_edge(u, v, cap):
        """Add directed edge u->v with capacity cap, plus reverse edge."""
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])
    
    # Add edges: source -> v_in (cap 2), v_out -> sink (cap 2)
    for v in range(n):
        add_edge(src, 2 * v, 2)       # source to v_in
        add_edge(2 * v + 1, sink, 2)  # v_out to sink
    
    # For each undirected edge (u,v), add u_out->v_in and v_out->u_in with cap 1
    edge_flow_map = {}  # (u,v) sorted -> flow value
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # u_out -> v_in
                idx_uv = len(graph[2*u+1])
                add_edge(2 * u + 1, 2 * v, 1)
                edge_flow_map[(u, v)] = (2*u+1, 2*v, idx_uv)
                # v_out -> u_in  
                idx_vu = len(graph[2*v+1])
                add_edge(2 * v + 1, 2 * u, 1)
                edge_flow_map[(v, u)] = (2*v+1, 2*u, idx_vu)
    
    # Edmonds-Karp max flow
    def bfs_flow():
        """Find augmenting path using BFS. Returns path or None."""
        parent = [-1] * total_nodes
        parent[src] = src
        q = deque([src])
        while q:
            u = q.popleft()
            for i, (v, cap, rev_idx) in enumerate(graph[u]):
                if cap > 0 and parent[v] == -1:
                    parent[v] = u
                    if v == sink:
                        # Reconstruct path
                        path = []
                        cur = v
                        while cur != src:
                            prev = parent[cur]
                            # Find the edge index in graph[prev] that goes to cur
                            for j, (vv, cc, rr) in enumerate(graph[prev]):
                                if vv == cur:
                                    path.append((prev, j))
                                    break
                            cur = prev
                        return path
                    q.append(v)
        return None
    
    max_flow = 0
    while True:
        path = bfs_flow()
        if path is None:
            break
        # Find min capacity along path
        min_cap = float('inf')
        for (u, idx) in path:
            min_cap = min(min_cap, graph[u][idx][1])
        # Augment
        for (u, idx) in path:
            graph[u][idx][1] -= min_cap
            v = graph[u][idx][0]
            rev_idx = graph[u][idx][2]
            graph[v][rev_idx][1] += min_cap
        max_flow += min_cap
    
    # Extract edges with flow > 0 from the 2-matching
    matching_edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # Check u_out -> v_in
                uo = 2 * u + 1
                vi = 2 * v
                for edge_data in graph[uo]:
                    if edge_data[0] == vi and edge_data[1] == 0:
                        # Original cap was 1, now 0 means flow 1
                        matching_edges.add((u, v))
                        break
    
    # Now extract cycles from the 2-matching subgraph
    # Build adjacency for matching subgraph
    match_adj = {v: [] for v in range(n)}
    for (u, v) in matching_edges:
        match_adj[u].append(v)
        match_adj[v].append(u)
    
    # Find cycles in the 2-matching (each vertex has degree 0, 1, or 2)
    cycles = []
    visited = set()
    for v in range(n):
        if v not in visited and len(match_adj[v]) == 2:
            # Start a cycle
            cycle = [v]
            visited.add(v)
            prev = -1
            cur = v
            while True:
                # Find next vertex
                nexts = match_adj[cur]
                if prev == -1:
                    nxt = nexts[0]
                else:
                    nxt = nexts[0] if nexts[0] != prev else nexts[1]
                if nxt == v:
                    break
                if nxt in visited:
                    break
                cycle.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
            if len(cycle) >= 3:
                cycles.append(cycle)
    
    return cycles, matching_edges

def _decompose(edges, n):
    """Decompose edges into cycles and single edges using iterative 2-matching."""
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    max_iterations = 50
    iters = 0
    while remaining and iters < max_iterations:
        # Find maximum 2-matching
        cycles, matching_edges = _max_2matching(adj, n)
        
        if not cycles:
            break
        
        # Remove all matching edges from remaining
        for cycle in cycles:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                if edge in remaining:
                    remaining.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
                pieces.append(cycle[:])
        
        iters += 1
    
    # Remaining edges become single-edge pieces
    for u, v in sorted(remaining):
        pieces.append([u, v])
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Return what we have so far, fill rest with singletons
            while len(results) < 3:
                s = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        pieces = _decompose(edges, n)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
