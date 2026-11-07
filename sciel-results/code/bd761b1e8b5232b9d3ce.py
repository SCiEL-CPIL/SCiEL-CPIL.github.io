"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _adj_matrix(edges, n):
    """Build adjacency set from edge list."""
    adj = set()
    for u, v in edges:
        adj.add((u, v))
        adj.add((v, u))
    return adj


def _is_triangle_free(adj, n):
    """Check if graph is triangle-free."""
    for u in range(n):
        for v in adj:
            if v[0] == u:
                for w in adj:
                    if w[0] == v[1] and w[1] > v[1]:
                        if (v[1], w[1]) in adj or (w[1], v[1]) in adj:
                            if u != v[1] and u != w[1]:
                                return False
    return True


def _bipartite_check(adj, n):
    """Check if graph is bipartite (necessary for triangle-free)."""
    color = [-1] * n
    for start in range(n):
        if color[start] == -1:
            color[start] = 0
            stack = [start]
            while stack:
                u = stack.pop()
                for v in adj:
                    if v[0] == u:
                        w = v[1]
                        if color[w] == -1:
                            color[w] = 1 - color[u]
                            stack.append(w)
                        elif color[w] == color[u]:
                            return False
    return True


def _diameter(adj, n):
    """Compute graph diameter via BFS from each node."""
    from collections import deque
    
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj:
                if v[0] == u:
                    w = v[1]
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        queue.append(w)
                        max_dist = max(max_dist, dist[w])
        # Check connectivity
        if -1 in dist:
            return float('inf')
    return max_dist


def _validate(added_edges, base_edges, n):
    """Check all constraints: no duplicates, triangle-free, connected, diameter <= 4."""
    # Check for duplicate or existing edges
    existing = set()
    for e in base_edges:
        existing.add(frozenset(e))
    for e in added_edges:
        fs = frozenset(e)
        if fs in existing:
            return False
        existing.add(fs)
    
    # Build adjacency
    adj = _adj_matrix(base_edges + added_edges, n)
    
    # Triangle-free check (bipartite is sufficient for this structure)
    if not _bipartite_check(adj, n):
        return False
    
    # Diameter check
    if _diameter(adj, n) > 4:
        return False
    
    return True


def construct():
    deadline = time.time() + 80  # Leave safety margin before 90s kill
    
    n = P['n']
    base_edges = P['edges']
    base_adj = _adj_matrix(base_edges, n)
    
    # All possible edges not in base
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_adj and (v, u) not in base_adj:
                all_possible.append((u, v))
    
    # Start with the bipartite complete graph between even and odd vertices
    # (this is the seed's approach)
    initial_added = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_adj and (v, u) not in base_adj:
                initial_added.append([u, v])
    
    # Verify initial solution is valid
    if not _validate(initial_added, base_edges, n):
        # Fallback: try to find any valid solution
        # Use greedy: add edges that reduce diameter while maintaining triangle-free
        added = []
        current_edges = list(base_edges)
        for _ in range(50):
            if time.time() > deadline:
                break
            cur_adj = _adj_matrix(current_edges, n)
            diam = _diameter(cur_adj, n)
            if diam <= 4 and _bipartite_check(cur_adj, n):
                break
            # Find the pair with max distance
            best_pair = None
            best_dist = 0
            from collections import deque
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    for v in cur_adj:
                        if v[0] == u:
                            w = v[1]
                            if dist[w] == -1:
                                dist[w] = dist[u] + 1
                                queue.append(w)
                for end in range(n):
                    if dist[end] > best_dist:
                        best_dist = dist[end]
                        best_pair = (start, end)
            
            if best_pair is None:
                break
            
            u, v = best_pair
            # Check if adding (u,v) keeps triangle-free
            new_edges = current_edges + [[u, v]]
            new_adj = _adj_matrix(new_edges, n)
            if _bipartite_check(new_adj, n):
                current_edges = new_edges
                added.append([u, v])
        
        if _validate(added, base_edges, n):
            initial_added = added
        else:
            # Last resort: return empty or minimal
            initial_added = []
    
    best_added = initial_added
    best_count = len(best_added)
    
    # Try to improve by removing edges
    # Sort added edges; try removing each one if valid
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_added)):
            if time.time() > deadline:
                break
            candidate = best_added[:i] + best_added[i+1:]
            if _validate(candidate, base_edges, n):
                best_added = candidate
                best_count = len(best_added)
                improved = True
                break
    
    # Try replacing pairs: remove one, add another
    if time.time() < deadline - 10:
        for i in range(len(best_added)):
            if time.time() > deadline - 10:
                break
            for edge in all_possible:
                if edge in best_added:
                    continue
                candidate = best_added[:i] + best_added[i+1:]
                candidate.append(list(edge))
                if _validate(candidate, base_edges, n):
                    # Same count but maybe enables further reduction
                    # Only keep if it allows removing another edge
                    for j in range(len(candidate)):
                        if j == len(candidate) - 1:
                            continue
                        if candidate[j] == list(edge):
                            continue
                        test = candidate[:j] + candidate[j+1:]
                        if _validate(test, base_edges, n):
                            best_added = test
                            best_count = len(best_added)
                            improved = True
                            break
                    if improved:
                        break
    
    # Final cleanup: try removing again
    if time.time() < deadline - 5:
        improved2 = True
        while improved2 and time.time() < deadline - 5:
            improved2 = False
            for i in range(len(best_added)):
                candidate = best_added[:i] + best_added[i+1:]
                if _validate(candidate, base_edges, n):
                    best_added = candidate
                    best_count = len(best_added)
                    improved2 = True
                    break
    
    # Ensure proper types
    result = []
    for e in best_added:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
