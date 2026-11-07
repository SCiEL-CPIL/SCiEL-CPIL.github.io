import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        existing_edges.add((min(u,v), max(u,v)))
    
    added_edges = []
    added_set = set()
    
    def get_diameter_and_farthest():
        max_diam = 0
        far_pairs = []
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if -1 in dist:
                return float('inf'), []
            local_max = max(dist)
            if local_max > max_diam:
                max_diam = local_max
                far_pairs = [(start, end) for end in range(n) if dist[end] == local_max]
            elif local_max == max_diam:
                far_pairs.extend([(start, end) for end in range(n) if dist[end] == local_max])
        return max_diam, far_pairs

    def is_valid(u, v):
        if u == v:
            return False
        u, v = min(u,v), max(u,v)
        if (u,v) in existing_edges or (u,v) in added_set:
            return False
        # Check triangle-free: u and v must not share a neighbor
        if adj[u] & adj[v]:
            return False
        return True

    def add_edge(u, v):
        u, v = min(u,v), max(u,v)
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append([int(u), int(v)])
        added_set.add((u,v))
        
    def remove_edge(u, v):
        u, v = min(u,v), max(u,v)
        adj[u].remove(v)
        adj[v].remove(u)
        # Remove from added_edges and added_set
        # added_edges is a list of lists, find and remove
        for i, e in enumerate(added_edges):
            if (e[0], e[1]) == (u, v) or (e[1], e[0]) == (u, v):
                added_edges.pop(i)
                break
        added_set.remove((u,v))

    deadline = time.time() + 55.0
    
    # Phase 1: Greedy Addition
    while time.time() < deadline:
        current_diam, far_pairs = get_diameter_and_farthest()
        if current_diam <= 4:
            break
        
        best_edge = None
        best_score = (float('inf'), float('inf')) # (count_of_pairs_gt_4, max_diam)
        
        # Precompute distances
        all_dists = []
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            all_dists.append(dist)
            
        # Count pairs with dist > 4
        pairs_gt_4 = 0
        for i in range(n):
            for j in range(i+1, n):
                if all_dists[i][j] > 4:
                    pairs_gt_4 += 1
                    
        # Try all valid edges
        for u in range(n):
            for v in range(u + 1, n):
                if not is_valid(u, v):
                    continue
                
                # Estimate new pairs > 4
                # We can approximate by checking if the new edge helps the far pairs
                # But for accuracy, we might need to update distances.
                # Since n=24, updating all distances is O(n^2) which is fine.
                
                # Temporarily add
                adj[u].add(v)
                adj[v].add(u)
                added_set.add((min(u,v), max(u,v)))
                
                # Calculate new diameter and pairs > 4
                new_diam = 0
                new_pairs_gt_4 = 0
                valid = True
                for start in range(n):
                    dist = [-1] * n
                    dist[start] = 0
                    queue = deque([start])
                    while queue:
                        node = queue.popleft()
                        for nb in adj[node]:
                            if dist[nb] == -1:
                                dist[nb] = dist[node] + 1
                                queue.append(nb)
                    if -1 in dist:
                        valid = False
                        break
                    local_max = max(dist)
                    if local_max > new_diam:
                        new_diam = local_max
                    for end in range(start+1, n):
                        if dist[end] > 4:
                            new_pairs_gt_4 += 1
                            
                # Undo
                adj[u].remove(v)
                adj[v].remove(u)
                added_set.remove((min(u,v), max(u,v)))
                
                if not valid:
                    continue
                    
                score = (new_pairs_gt_4, new_diam)
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
                    
        if best_edge is None:
            break
            
        add_edge(best_edge[0], best_edge[1])
        
    # Phase 2: Local Search (Removal)
    # Try to remove edges while maintaining validity
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(added_edges) - 1, -1, -1):
            u, v = added_edges[i]
            # Try removing
            adj[u].remove(v)
            adj[v].remove(u)
            added_set.remove((min(u,v), max(u,v)))
            
            # Check validity
            current_diam, _ = get_diameter_and_farthest()
            if current_diam <= 4:
                # Valid, keep removed
                added_edges.pop(i)
                improved = True
            else:
                # Invalid, add back
                adj[u].add(v)
                adj[v].add(u)
                added_set.add((min(u,v), max(u,v)))
                # Re-insert at position i? Or just append?
                # Since we iterate backwards, popping and not adding back is fine.
                # But we need to restore state if we add back.
                # The list order doesn't matter for the graph, but for consistency:
                added_edges.insert(i, [u, v])

    return [[int(u), int(v)] for u, v in added_edges]
