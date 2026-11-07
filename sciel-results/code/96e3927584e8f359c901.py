import time
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_START_TIME = time.time()
_TIME_LIMIT = 80.0

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _bfs(adj, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _get_diameter(adj, n):
    max_d = 0
    for i in range(n):
        dist = _bfs(adj, n, i)
        if -1 in dist:
            return 999
        max_d = max(max_d, max(dist))
    return max_d

def _is_triangle_free_add(adj, n, u, v):
    # Check if adding (u,v) creates a triangle
    # This happens if there exists w such that (u,w) in adj and (v,w) in adj
    for w in adj[u]:
        if w != v and w in adj[v]:
            return False
    return True

def _count_pairs_over_diameter(adj, n, limit):
    count = 0
    for i in range(n):
        dist = _bfs(adj, n, i)
        for j in range(i+1, n):
            if dist[j] > limit:
                count += 1
    return count

def construct():
    n = P['n']
    base_edges = P['edges']
    
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
        
    added_edges = []
    
    # Generate all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                all_edges.append((i, j))
                
    # Greedy Phase
    # Prioritize bipartite edges (i%2 != j%2) as they are always triangle-free
    bipartite_edges = [e for e in all_edges if e[0]%2 != e[1]%2]
    non_bipartite_edges = [e for e in all_edges if e[0]%2 == e[1]%2]
    
    # Try greedy with bipartite edges first
    candidates = bipartite_edges
    
    while _time_ok():
        pairs_over = _count_pairs_over_diameter(adj, n, 4)
        if pairs_over == 0:
            break
            
        best_edge = None
        best_pairs = pairs_over
        
        for u, v in candidates:
            if (u, v) in edge_set:
                continue
            if not _is_triangle_free_add(adj, n, u, v):
                continue
                
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            
            new_pairs = _count_pairs_over_diameter(adj, n, 4)
            
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove((u, v))
            
            if new_pairs < best_pairs:
                best_pairs = new_pairs
                best_edge = (u, v)
                
        if best_edge is None:
            # If no bipartite edge helps, try non-bipartite?
            # But non-bipartite might create triangles.
            # We already check _is_triangle_free_add.
            # So we can just expand candidates to all_edges if stuck.
            candidates = all_edges
            continue
            
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        added_edges.append([u, v])
        
    # Local Search
    if _time_ok():
        improved = True
        while improved and _time_ok():
            improved = False
            # Try removing
            for i in range(len(added_edges)):
                if not _time_ok():
                    break
                u, v = added_edges[i]
                
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.remove((u, v))
                
                if _get_diameter(adj, n) <= 4:
                    added_edges.pop(i)
                    improved = True
                    break
                else:
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
                    
            # Try swapping (if removal failed)
            if not improved:
                # Pick a random added edge to remove, and a random candidate to add
                if added_edges:
                    idx = random.randint(0, len(added_edges)-1)
                    u, v = added_edges[idx]
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    edge_set.remove((u, v))
                    added_edges.pop(idx)
                    
                    # Find a good edge to add
                    best_new_edge = None
                    best_pairs = _count_pairs_over_diameter(adj, n, 4)
                    
                    # Sample candidates
                    sample = random.sample(all_edges, min(50, len(all_edges)))
                    for nu, nv in sample:
                        if (nu, nv) in edge_set:
                            continue
                        if not _is_triangle_free_add(adj, n, nu, nv):
                            continue
                            
                        adj[nu].add(nv)
                        adj[nv].add(nu)
                        edge_set.add((nu, nv))
                        
                        new_pairs = _count_pairs_over_diameter(adj, n, 4)
                        
                        adj[nu].remove(nv)
                        adj[nv].remove(nu)
                        edge_set.remove((nu, nv))
                        
                        if new_pairs < best_pairs:
                            best_pairs = new_pairs
                            best_new_edge = (nu, nv)
                            
                    if best_new_edge:
                        nu, nv = best_new_edge
                        adj[nu].add(nv)
                        adj[nv].add(nu)
                        edge_set.add((nu, nv))
                        added_edges.append([nu, nv])
                        improved = True
                    else:
                        # Revert
                        adj[u].add(v)
                        adj[v].add(u)
                        edge_set.add((u, v))
                        added_edges.insert(idx, [u, v])

    if _get_diameter(adj, n) > 4:
        return []
        
    return added_edges
