"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj, n, u, v):
    # Check if adding edge (u,v) creates a triangle
    for w in adj[u]:
        if w != v and w in adj[v]:
            return False
    return True

def _bfs_diameter(adj, n):
    # Compute diameter via BFS from each vertex
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        for i in range(len(queue)):
            u = queue[i]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d > max_dist:
                max_dist = d
    return max_dist

def _count_pairs_over4(adj, n, limit=4):
    # Count number of unordered pairs at distance > limit
    count = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        for i in range(len(queue)):
            u = queue[i]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for v in range(start + 1, n):
            if dist[v] > limit:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    base_adj = _build_adj(base_edges, n)
    
    # All possible edges not in base
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_possible.append((u, v))
    
    # Greedy: repeatedly add the edge that most reduces pairs-over-4,
    # subject to triangle-free constraint
    added_edges = []
    current_edges = set(base_edges)
    current_adj = _build_adj(current_edges, n)
    
    while True:
        if time.time() > deadline:
            break
        # Check if diameter is already <= 4
        diam = _bfs_diameter(current_adj, n)
        if diam <= 4:
            break
        
        best_edge = None
        best_score = float('inf')
        
        for (u, v) in all_possible:
            if (u, v) in current_edges:
                continue
            if not _is_triangle_free(current_adj, n, u, v):
                continue
            # Temporarily add edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            score = _count_pairs_over4(current_adj, n, 4)
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            
            if score < best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        current_edges.add((u, v))
        current_adj[u].add(v)
        current_adj[v].add(u)
        added_edges.append([u, v])
    
    # Local search: try to remove edges while maintaining diameter <= 4 and triangle-free
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(added_edges)):
            if time.time() > deadline:
                break
            u, v = added_edges[i]
            # Try removing this edge
            new_edges = set(current_edges)
            new_edges.discard((u, v))
            new_adj = _build_adj(new_edges, n)
            diam = _bfs_diameter(new_adj, n)
            if diam <= 4:
                current_edges = new_edges
                current_adj = new_adj
                added_edges.pop(i)
                improved = True
                break
    
    # Ensure return type is correct
    result = [[int(u), int(v)] for u, v in added_edges]
    return result
# EVOLVE-BLOCK-END

