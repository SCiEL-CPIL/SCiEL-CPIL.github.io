"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_distances(adj, start, n):
    dist = [-1] * n
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

def _diameter(adj, n):
    max_d = 0
    for s in range(n):
        dist = _bfs_distances(adj, s, n)
        for d in dist:
            if d > max_d:
                max_d = d
        if max_d > 4:
            return max_d
    return max_d

def _is_triangle_free(adj, n):
    for u in range(n):
        adj_u = adj[u]
        for v in adj_u:
            if v > u:
                common = adj_u & adj[v]
                if common:
                    return False
    return True

def _valid_solution(added_edges, base_edges, n):
    all_edges = list(base_edges) + added_edges
    edge_set = set()
    for u, v in all_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    adj = _build_adj(n, all_edges)
    if not _is_triangle_free(adj, n):
        return False
    diam = _diameter(adj, n)
    if diam > 4:
        return False
    return True

def _get_all_pairs_over_diam(adj, n, limit=4):
    """Get all pairs with distance > limit."""
    pairs = []
    for s in range(n):
        dist = _bfs_distances(adj, s, n)
        for t in range(s+1, n):
            if dist[t] > limit:
                pairs.append((s, t, dist[t]))
    return pairs

def _edge_covers_pair(u, v, s, t, adj, n):
    """Check if adding edge (u,v) reduces distance between s and t."""
    # This is complex; use BFS instead for accuracy
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = P['n']
    base_edges = P['edges']
    base_adj = _build_adj(n, base_edges)
    
    # Get candidate edges: same-parity, not adjacent, not distance 2 (would create triangle with path)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if abs(u - v) <= 2:
                continue  # too close, would create triangle or be base edge
            if (u % 2) != (v % 2):
                continue  # different parity would create odd cycle
            candidates.append((u, v))
    
    # Sort candidates by "usefulness" - prefer edges that connect distant vertices
    candidates.sort(key=lambda e: abs(e[0] - e[1]), reverse=True)
    
    # Known good solution: star from 12
    best = [[0, 12], [2, 12], [4, 12], [6, 12], [8, 12], [15, 12], [17, 12], [19, 12], [21, 12], [23, 12]]
    
    if _valid_solution(best, base_edges, n):
        pass  # confirmed valid
    else:
        # Fallback: full star from 12
        fallback = []
        for v in range(n):
            if v != 12 and abs(v - 12) > 2 and (v % 2) == (12 % 2):
                fallback.append([min(12, v), max(12, v)])
        best = fallback
    
    # Greedy set-cover approach
    # Start with empty set, greedily add edges that reduce diameter most
    current_edges = []
    current_adj = [set(base_adj[i]) for i in range(n)]
    
    # Compute initial distances
    def compute_max_dist(adj):
        max_d = 0
        for s in range(n):
            dist = _bfs_distances(adj, s, n)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    current_max = compute_max_dist(current_adj)
    
    # Greedy: add edges that most reduce the maximum distance
    available_candidates = list(candidates)
    
    while current_max > 4 and available_candidates and time.time() < deadline:
        best_edge = None
        best_max_after = current_max
        
        for edge in available_candidates:
            u, v = edge
            # Try adding this edge
            test_adj = [set(a) for a in current_adj]
            test_adj[u].add(v)
            test_adj[v].add(u)
            
            # Check triangle-free
            if not _is_triangle_free(test_adj, n):
                continue
            
            max_after = compute_max_dist(test_adj)
            if max_after < best_max_after:
                best_max_after = max_after
                best_edge = edge
        
        if best_edge is None:
            break
        
        u, v = best_edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        current_edges.append([u, v])
        available_candidates.remove(best_edge)
        current_max = best_max_after
    
    # Now try to remove redundant edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_edges)):
            trial = current_edges[:i] + current_edges[i+1:]
            if _valid_solution(trial, base_edges, n):
                current_edges = trial
                improved = True
                break
    
    if len(current_edges) < len(best):
        best = current_edges
    
    # Also try star-based solutions for comparison
    for center in range(4, 20):
        if time.time() > deadline:
            break
        star_edges = []
        for v in range(n):
            if v == center or abs(v - center) <= 2:
                continue
            if (v % 2) == (center % 2):
                star_edges.append([min(center, v), max(center, v)])
        
        if _valid_solution(star_edges, base_edges, n):
            # Try to reduce
            current = list(star_edges)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(current)):
                    trial = current[:i] + current[i+1:]
                    if _valid_solution(trial, base_edges, n):
                        current = trial
                        improved = True
                        break
            if len(current) < len(best):
                best = current
    
    # Final: try to find even better solutions with targeted search
    # Use the best solution as starting point and try local modifications
    if time.time() < deadline:
        # Try adding one edge and removing two
        for i in range(len(best)):
            for j in range(i+1, len(best)):
                if time.time() > deadline:
                    break
                reduced = [e for k, e in enumerate(best) if k != i and k != j]
                # Try adding a new edge
                for cand in candidates[:50]:  # top candidates
                    u, v = cand
                    if [min(u,v), max(u,v)] in reduced:
                        continue
                    trial = reduced + [[min(u,v), max(u,v)]]
                    if _valid_solution(trial, base_edges, n):
                        best = trial
                        break
                if time.time() > deadline:
                    break
    
    # Convert to proper types
    result = []
    for e in best:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
