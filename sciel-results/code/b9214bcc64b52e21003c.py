"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs(adj, src, n):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist

def _check_solution(added_edges, n):
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check triangle-free: for each vertex, its neighbors must form an independent set
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    return False
    
    # Check connected and diameter <= 4
    for i in range(n):
        d = _bfs(adj, i, n)
        for j in range(n):
            if d[j] == -1 or d[j] > 4:
                return False
    return True

def _diameter(adj, n):
    diam = 0
    for i in range(n):
        d = _bfs(adj, i, n)
        for j in range(n):
            if d[j] == -1:
                return -1  # disconnected
            diam = max(diam, d[j])
    return diam

def construct():
    start_time = time.time()
    n = P['n']
    deadline = 85
    
    # Precompute path edges
    path_edges = set()
    for u, v in P['edges']:
        path_edges.add((min(u, v), max(u, v)))
    
    even_vertices = list(range(0, n, 2))  # 0,2,4,...,22
    odd_vertices = list(range(1, n, 2))   # 1,3,5,...,23
    
    # All valid candidate edges (even-odd, not in path)
    candidates = []
    for e in even_vertices:
        for o in odd_vertices:
            pair = (min(e, o), max(e, o))
            if pair not in path_edges:
                candidates.append(pair)
    
    # Key insight: start with a valid solution, then remove edges greedily
    # Start with a hub-and-spoke solution that's guaranteed to work
    
    def _build_adj_from_edge_set(edge_set, n):
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def _is_valid(edge_list, n):
        if time.time() - start_time > deadline:
            return None
        return _check_solution(edge_list, n)
    
    # Strategy: Start from a known valid solution and remove edges
    # First, find a valid solution using a star pattern
    
    # Star pattern: connect each even vertex to nearby odd vertices
    # Or better: use a "ladder" pattern
    
    # Let's try: for each odd vertex, connect to the closest even vertex within distance 4
    # Actually, let's start with a dense valid solution and remove
    
    # Initial dense solution: connect every other vertex to create shortcuts
    # Pattern: add edges (i, i+5) for appropriate parity
    # But let's just start with all edges that reduce diameter and prune
    
    # Better approach: Start with a solution that definitely works
    # Connect vertex 0 to {5, 10, 15, 20} and vertex 23 to {18, 13, 8, 3}
    # This creates "highways" at both ends
    
    # Actually, let me think about what's needed:
    # Path 0-1-2-...-23 has diameter 23
    # We need diameter <= 4, so max distance between any two vertices is 4
    
    # A bipartite graph with 12 even and 12 odd vertices
    # Each even vertex has degree at most... well, it can connect to any odd vertex
    # But no triangles means if even e connects to odd o1 and o2, then o1 and o2 can't both connect to any common even
    
    # Wait, that's not right. Triangle-free means no three vertices all mutually connected.
    # In a bipartite graph, triangles can only form if... wait, bipartite graphs are automatically triangle-free!
    # Because any cycle in a bipartite graph has even length, so no 3-cycles.
    
    # So ANY set of even-odd edges added to the path will be triangle-free!
    # Wait, but the path itself has edges between consecutive vertices (even-odd).
    # The path is already bipartite (even-odd coloring).
    # So adding any even-odd edges keeps it bipartite, hence triangle-free.
    
    # This is a KEY insight: the triangle-free constraint is automatically satisfied
    # because the graph remains bipartite with the natural even/odd partition.
    
    # So the problem reduces to: find minimum number of even-odd edges to add
    # so that the resulting graph has diameter <= 4.
    
    # Let's use a removal-based approach:
    # 1. Start with a valid (but not minimal) solution
    # 2. Try to remove edges while maintaining validity
    
    # Find an initial valid solution:
    # Connect every even vertex to a few odd vertices to create shortcuts
    
    # Simple valid solution: create a "complete bipartite" between two sets
    # Or: connect vertex 6 to {0,12} and vertex 12 to {6,18} etc.
    
    # Let's just try: add edges to make it a small-world network
    # Each vertex should be within 4 of every other
    
    # Greedy construction:
    # Start with empty added edges
    # Repeatedly add the edge that reduces the maximum distance the most
    
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def _get_diameter(adj, n):
        max_d = 0
        for i in range(n):
            d = _bfs(adj, i, n)
            for j in range(n):
                if d[j] == -1:
                    return 999
                max_d = max(max_d, d[j])
        return max_d
    
    current_edges = []
    cur_diam = _get_diameter(adj, n)
    
    # Greedy: add edges that most reduce diameter
    while cur_diam > 4 and time.time() - start_time < deadline:
        best_edge = None
        best_new_diam = cur_diam + 1
        
        for u, v in candidates:
            if (u, v) in set(map(tuple, current_edges)):
                continue
            if (v, u) in set(map(tuple, current_edges)):
                continue
            
            # Temporarily add edge
            adj[u].add(v)
            adj[v].add(u)
            
            new_diam = _get_diameter(adj, n)
            
            # Remove edge
            adj[u].discard(v)
            adj[v].discard(u)
            
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (u, v)
        
        if best_edge is None or best_new_diam >= cur_diam:
            break
        
        current_edges.append(list(best_edge))
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        cur_diam = best_new_diam
    
    # Now try to remove edges
    if time.time() - start_time < deadline:
        improved = True
        while improved and time.time() - start_time < deadline:
            improved = False
            for i in range(len(current_edges)):
                if time.time() - start_time > deadline:
                    break
                # Try removing edge i
                trial = [e for j, e in enumerate(current_edges) if j != i]
                if _is_valid(trial, n):
                    current_edges = trial
                    improved = True
                    break  # restart the loop since indices changed
    
    if current_edges and _is_valid(current_edges, n):
        return [[int(u), int(v)] for u, v in current_edges]
    
    # Fallback: hub-based solution
    # Connect a central hub to all far vertices
    hub = 11  # odd vertex
    fallback = []
    for v in range(n):
        if v == hub:
            continue
        d = _bfs(adj, hub, n)
        if d[v] > 4:
            # Need to connect v to something closer to hub
            # Just connect v to hub if bipartite-compatible
            # hub is odd (11), so v must be even
            if v % 2 == 0:
                pair = (min(hub, v), max(hub, v))
                if pair not in path_edges and pair not in set(map(tuple, fallback)):
                    fallback.append([hub, v])
    
    # If fallback doesn't work, use a more aggressive strategy
    if not _is_valid(fallback, n):
        # Use multiple hubs
        fallback = []
        hubs = [5, 10, 15, 20]  # even vertices as hubs
        for h in hubs:
            for v in range(n):
                if v == h or v % 2 == h % 2:
                    continue  # must be opposite parity
                pair = (min(h, v), max(h, v))
                if pair not in path_edges and pair not in set(map(tuple, fallback)):
                    fallback.append([h, v])
    
    return [[int(u), int(v)] for u, v in fallback]
# EVOLVE-BLOCK-END
