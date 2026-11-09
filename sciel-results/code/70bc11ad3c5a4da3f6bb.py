"""Triangle-free diameter-four augmentation via direct construction."""
# EVOLVE-BLOCK-START

import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _validate(added_edges):
    """Validate the added edges: triangle-free, connected, diameter <= 4, no duplicates, no existing edges."""
    n = 24
    base_edge_set = set()
    for u, v in P['edges']:
        base_edge_set.add((min(u, v), max(u, v)))
    
    # Check no duplicate added edges, no existing edges, valid vertices
    added_edge_set = set()
    for u, v in added_edges:
        if not (isinstance(u, int) and isinstance(v, int)):
            return False
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in base_edge_set or key in added_edge_set:
            return False
        added_edge_set.add(key)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check triangle-free
    for a in range(n):
        for b in adj[a]:
            if b > a:
                for c in adj[b]:
                    if c > b and c in adj[a]:
                        return False
    
    # BFS diameter check
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 4:
                        return False
                    queue.append(v)
        for i in range(n):
            if dist[i] == -1:
                return False
    
    return True


def construct():
    """
    Triangle-free diameter-four augmentation.
    
    The base graph is a path on 24 vertices. All added edges must be between
    even and odd vertices (to maintain triangle-free/bipartite property).
    
    Strategy:
    1. Return known 10-edge solution immediately
    2. Try to find better solutions if time permits
    """
    t0 = time.time()
    deadline = t0 + 85  # 85s safety margin (limit is 90s)
    n = 24
    
    # Known good solution: hub 12 connects to all odd vertices except 11,13
    fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    
    best = [list(e) for e in fallback]
    best_val = len(best)
    
    # Fast validation helper that uses precomputed base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
    base_edge_set = set((min(u,v), max(u,v)) for u, v in P['edges'])
    
    even_verts = list(range(0, n, 2))  # 0,2,4,...,22
    odd_verts = list(range(1, n, 2))   # 1,3,5,...,23
    
    # All valid added edges (even-odd pairs not in base path)
    all_possible_edges = []
    for e in even_verts:
        for o in odd_verts:
            key = (min(e,o), max(e,o))
            if key not in base_edge_set:
                all_possible_edges.append([e, o])
    
    def fast_validate(added_edges):
        """Fast validation: triangle-free, connected, diameter <= 4."""
        # Check for duplicates and existing edges
        seen = set()
        for u, v in added_edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u,v), max(u,v))
            if key in base_edge_set or key in seen:
                return False
            seen.add(key)
        
        # Build adjacency
        adj = [set(base_adj[i]) for i in range(n)]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Triangle-free check
        for a in range(n):
            for b in adj[a]:
                if b > a:
                    for c in adj[b]:
                        if c > b and c in adj[a]:
                            return False
        
        # BFS diameter check (early exit if > 4)
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
            for i in range(n):
                if dist[i] == -1:
                    return False
        
        return True
    
    # Phase 1: Try removing single edges from best
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            trial = best[:i] + best[i+1:]
            if fast_validate(trial):
                best = trial
                best_val = len(best)
                changed = True
                break
    
    # Phase 2: Try removing pairs of edges
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(best)):
                if time.time() > deadline:
                    break
                trial = best[:i] + best[i+1:j] + best[j+1:]
                if fast_validate(trial):
                    best = trial
                    best_val = len(best)
                    changed = True
                    break
            if changed:
                break
    
    # Phase 3: 1-swap local search (remove one edge, add one different edge)
    if time.time() < deadline:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            remaining = best[:i] + best[i+1:]
            for alt in all_possible_edges:
                if time.time() > deadline:
                    break
                if alt in remaining:
                    continue
                trial = remaining + [alt]
                if len(trial) < best_val and fast_validate(trial):
                    best = trial
                    best_val = len(best)
                    break
            if len(best) < best_val:
                break
    
    # Phase 4: 2-swap local search (remove two edges, add two different edges)
    if time.time() < deadline:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(best)):
                if time.time() > deadline:
                    break
                remaining = best[:i] + best[i+1:j] + best[j+1:]
                for alt1 in all_possible_edges:
                    if time.time() > deadline:
                        break
                    if alt1 in remaining:
                        continue
                    for alt2 in all_possible_edges:
                        if time.time() > deadline:
                            break
                        if alt2 in remaining or alt2 == alt1:
                            continue
                        trial = remaining + [alt1, alt2]
                        if len(trial) < best_val and fast_validate(trial):
                            best = trial
                            best_val = len(best)
                            break
                    if len(best) < best_val:
                        break
                if len(best) < best_val:
                    break
            if len(best) < best_val:
                break
    
    # Phase 5: Try structured hub configurations
    if time.time() < deadline:
        # Try single hub at various positions
        for hub in even_verts:
            if time.time() > deadline:
                break
            # Hub connects to odd vertices
            needed_odds = []
            for o in odd_verts:
                d = abs(o - hub)
                if d > 2:
                    needed_odds.append(o)
            config = [[hub, o] for o in needed_odds]
            if len(config) < best_val and fast_validate(config):
                best = config
                best_val = len(best)
        
        for hub in odd_verts:
            if time.time() > deadline:
                break
            needed_evens = []
            for e in even_verts:
                d = abs(e - hub)
                if d > 2:
                    needed_evens.append(e)
            config = [[hub, e] for e in needed_evens]
            if len(config) < best_val and fast_validate(config):
                best = config
                best_val = len(best)
    
    # Phase 6: Try two-hub configurations
    if time.time() < deadline:
        for h1 in even_verts:
            if time.time() > deadline:
                break
            for h2 in odd_verts:
                if time.time() > deadline:
                    break
                edges = []
                if (min(h1,h2), max(h1,h2)) not in base_edge_set:
                    edges.append([h1, h2])
                
                for o in odd_verts:
                    if abs(o - h1) > 2:
                        edges.append([h1, o])
                for e in even_verts:
                    if abs(e - h2) > 2:
                        edges.append([h2, e])
                
                if len(edges) < best_val and fast_validate(edges):
                    best = [list(e) for e in edges]
                    best_val = len(best)
    
    # Final validation and cleanup
    if not fast_validate(best):
        best = [list(e) for e in fallback]
    
    result = []
    for edge in best:
        result.append([int(edge[0]), int(edge[1])])
    
    return result

# EVOLVE-BLOCK-END
