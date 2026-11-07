import time
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 60s limit, keep 5s margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    # Candidate edges: Bipartite (different parity) and not in base
    # This guarantees triangle-freeness since all added edges cross parity
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 != 0:  # Different parity
                if (u, v) not in base_edges:
                    candidates.append((u, v))
    
    # Initial graph state using adjacency lists
    adj = {i: [] for i in range(n)}
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    added_edges = []
    
    def compute_distances():
        """Compute all-pairs shortest path distances using BFS from each node."""
        dists = [[-1] * n for _ in range(n)]
        for i in range(n):
            queue = [i]
            dists[i][i] = 0
            while queue:
                next_queue = []
                for node in queue:
                    for neighbor in adj[node]:
                        if dists[i][neighbor] == -1:
                            dists[i][neighbor] = dists[i][node] + 1
                            next_queue.append(neighbor)
                queue = next_queue
        return dists

    def get_max_and_sum(dists):
        max_d = 0
        sum_d = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i][j] > max_d:
                    max_d = dists[i][j]
                sum_d += dists[i][j]
        return max_d, sum_d

    def get_bad_pairs(dists):
        """Count pairs with distance > 4."""
        bad = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i][j] > 4:
                    bad += 1
        return bad

    def is_edge_added(u, v):
        for x in added_edges:
            if (x[0] == u and x[1] == v) or (x[0] == v and x[1] == u):
                return True
        return False

    # Greedy Loop: Add edges that most reduce bad pairs, then diameter, then sum
    while True:
        if time.time() > deadline:
            break
            
        dists = compute_distances()
        max_d, sum_d = get_max_and_sum(dists)
        bad_count = get_bad_pairs(dists)
        
        if max_d <= 4:
            break
            
        best_edge = None
        best_score = None
        
        # Evaluate each candidate edge
        for e in candidates:
            u, v = e
            if (u, v) in base_edges:
                continue
            if is_edge_added(u, v):
                continue
                
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            
            new_dists = compute_distances()
            new_max, new_sum = get_max_and_sum(new_dists)
            new_bad = get_bad_pairs(new_dists)
            
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Score: Minimize bad pairs first, then max_d, then sum_d
            score = (new_bad, new_max, new_sum)
            
            if best_score is None or score < best_score:
                best_score = score
                best_edge = e
        
        if best_edge is None:
            break
            
        # Add the best edge
        u, v = best_edge
        added_edges.append([u, v])
        adj[u].append(v)
        adj[v].append(u)
        
        # Remove from candidates to avoid re-considering
        try:
            candidates.remove(best_edge)
        except ValueError:
            pass
    
    # Local search: Try to remove edges while maintaining diameter <= 4
    removed_any = True
    while removed_any:
        if time.time() > deadline:
            break
        removed_any = False
        
        for i in range(len(added_edges) - 1, -1, -1):
            if time.time() > deadline:
                break
            u, v = added_edges[i]
            
            # Temporarily remove
            adj[u].remove(v)
            adj[v].remove(u)
            
            dists = compute_distances()
            max_d, _ = get_max_and_sum(dists)
            
            if max_d <= 4:
                # Keep it removed
                added_edges.pop(i)
                removed_any = True
            else:
                # Put it back
                adj[u].append(v)
                adj[v].append(u)
    
    # Final verification
    dists = compute_distances()
    max_d, _ = get_max_and_sum(dists)
    
    if max_d > 4:
        # Fallback: this shouldn't happen, but if it does, try to fix
        # by adding more edges greedily until diameter <= 4
        while max_d > 4 and time.time() < deadline:
            best_edge = None
            best_score = None
            for e in candidates:
                u, v = e
                if (u, v) in base_edges or is_edge_added(u, v):
                    continue
                adj[u].append(v)
                adj[v].append(u)
                new_dists = compute_distances()
                new_max, new_sum = get_max_and_sum(new_dists)
                adj[u].remove(v)
                adj[v].remove(u)
                score = (new_max, new_sum)
                if best_score is None or score < best_score:
                    best_score = score
                    best_edge = e
            if best_edge is None:
                break
            u, v = best_edge
            added_edges.append([u, v])
            adj[u].append(v)
            adj[v].append(u)
            dists = compute_distances()
            max_d, _ = get_max_and_sum(dists)
    
    return [[int(u), int(v)] for u, v in added_edges]
# EVOLVE-BLOCK-END
