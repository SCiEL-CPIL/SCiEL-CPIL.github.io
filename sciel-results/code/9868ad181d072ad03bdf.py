"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edge_set = set()
    for e in P['edges']:
        base_edge_set.add((e[0], e[1]))
        base_edge_set.add((e[1], e[0]))
    
    # Adjacency list
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate edges: (u,v) with v-u >= 3 (to avoid triangles in path)
    # Actually, we need to check triangle-freeness dynamically.
    
    # For a path, adding (u,v) with v-u=2 creates triangle u, u+1, u+2.
    # Adding (u,v) with v-u>=3 is safe initially, but after other edges are added,
    # we need to recheck.
    
    # Let's use a different approach: build the graph incrementally.
    # Use BFS to compute distances.
    
    def bfs_distances(adj, n, source):
        dist = [-1] * n
        dist[source] = 0
        queue = [source]
        idx = 0
        while idx < len(queue):
            u = queue[idx]
            idx += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist
    
    def get_diameter(adj, n):
        max_d = 0
        for s in range(n):
            dist = bfs_distances(adj, n, s)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    def can_add_edge(adj, u, v, n):
        # Check if adding (u,v) keeps graph triangle-free
        # Check common neighbors
        for w in adj[u]:
            if w != v and w in adj[v]:
                return False
        return True
    
    # Greedy: add edges that most reduce the diameter
    current_adj = [set(s) for s in adj]
    added = []
    
    # Precompute all candidate edges
    all_candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u,v) not in base_edge_set:
                all_candidates.append((u,v))
    
    while True:
        diam = get_diameter(current_adj, n)
        if diam <= 4:
            break
        
        best_edge = None
        best_diam = n
        for (u,v) in all_candidates:
            if (u,v) in base_edge_set or (v,u) in base_edge_set:
                continue
            if u in current_adj[v]:  # already added
                continue
            if not can_add_edge(current_adj, u, v, n):
                continue
            # Try adding
            current_adj[u].add(v)
            current_adj[v].add(u)
            new_diam = get_diameter(current_adj, n)
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u,v)
        
        if best_edge is None:
            break
        u, v = best_edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        added.append([u,v])
    
    # Try to remove edges
    changed = True
    while changed:
        changed = False
        for i in range(len(added)):
            u, v = added[i]
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            diam = get_diameter(current_adj, n)
            if diam <= 4:
                added.pop(i)
                changed = True
                break
            else:
                current_adj[u].add(v)
                current_adj[v].add(u)
    
    return added
# EVOLVE-BLOCK-END
