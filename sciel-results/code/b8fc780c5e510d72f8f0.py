import sys
import itertools
from collections import deque

def construct():
    P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}
    n = P['n']
    base_edges = set(tuple(e) for e in P['edges'])
    base_edges |= set((v, u) for u, v in base_edges)
    
    # All valid cross-parity edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 != 0:
                if (u, v) not in base_edges:
                    all_edges.append((u, v))
                    
    added_edges = set()
    
    def get_adj():
        adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bfs_diameter(adj):
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            # Check connectivity
            if -1 in dist:
                return float('inf')
            max_dist = max(max_dist, max(dist))
        return max_dist

    # Greedy addition
    while True:
        adj = get_adj()
        diam = bfs_diameter(adj)
        if diam <= 4:
            break
            
        # Find pair with max distance
        # To save time, we can just pick the first pair with distance > 4
        # Or the one with max distance
        target_dist = 0
        target_pair = None
        for u in range(n):
            for v in range(u + 1, n):
                # Simple BFS from u to v? Too slow if done for all pairs.
                # We already computed full BFS in bfs_diameter? No, we only returned max.
                # Let's modify bfs_diameter to return distances or recompute.
                pass
        
        # Optimization: Just try all candidate edges and see which reduces diameter most.
        best_edge = None
        best_diam = float('inf')
        
        # To speed up, we can limit candidates to those that are "far" apart?
        # Or just iterate all. 100+ edges, BFS is 24 nodes. 100 * 24 * 24 ~ 60k ops. Very fast.
        
        for e in all_edges:
            if e in added_edges:
                continue
            added_edges.add(e)
            adj = get_adj()
            d = bfs_diameter(adj)
            added_edges.remove(e)
            
            if d < best_diam:
                best_diam = d
                best_edge = e
                
        if best_edge is None:
            break
            
        added_edges.add(best_edge)
        
    # Pruning
    # Try to remove edges
    for e in list(added_edges):
        added_edges.remove(e)
        adj = get_adj()
        d = bfs_diameter(adj)
        if d > 4:
            added_edges.add(e)
            
    return [[u, v] for u, v in added_edges]
