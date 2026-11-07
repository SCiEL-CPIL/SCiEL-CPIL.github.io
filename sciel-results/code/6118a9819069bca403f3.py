import time
import random

P = {'n': 50, 'edges': [[0,1],[0,7],[0,43],[0,49],[1,2],[1,8],[1,44],[2,3],[2,9],[2,45],[3,4],[3,10],[3,46],[4,5],[4,11],[4,47],[5,6],[5,12],[5,48],[6,7],[6,13],[6,49],[7,8],[7,14],[8,9],[8,15],[9,10],[9,16],[10,11],[10,17],[11,12],[11,18],[12,13],[12,19],[13,14],[13,20],[14,15],[14,21],[15,16],[15,22],[16,17],[16,23],[17,18],[17,24],[18,19],[18,25],[19,20],[19,26],[20,21],[20,27],[21,22],[21,28],[22,23],[22,29],[23,24],[23,30],[24,25],[24,31],[25,26],[25,32],[26,27],[26,33],[27,28],[27,34],[28,29],[28,35],[29,30],[29,36],[30,31],[30,37],[31,32],[31,38],[32,33],[32,39],[33,34],[33,40],[34,35],[34,41],[35,36],[35,42],[36,37],[36,43],[37,38],[37,44],[38,39],[38,45],[39,40],[39,46],[40,41],[40,47],[41,42],[41,48],[42,43],[42,49],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49]]}

def construct():
    start = time.time()
    deadline = start + 80
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    best = None
    best_k = n + 1
    
    # Try k from 3 upward
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
        
        # Multiple random restarts for greedy
        found = False
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            colors = _greedy(n, adj_list, k, attempt)
            if colors is not None:
                # Try to improve with local search
                colors = _local_search(n, adj_list, colors, deadline)
                used = len(set(colors))
                if used < best_k:
                    best_k = used
                    best = list(colors)
                if k == 3 and best_k == 3:
                    return best
                found = True
                break
        
        if found and best is not None and best_k <= k:
            break
    
    if best is None:
        best = list(range(n))
    
    # Ensure Python int types
    return [int(x) for x in best]

def _greedy(n, adj_list, k, seed):
    """Greedy coloring with dynamic vertex ordering."""
    rng = random.Random(seed)
    colors = [-1] * n
    uncolored = set(range(n))
    
    # DSU store for bicolored pairs
    dsu_store = {}
    
    def dsu_find(dsu, x):
        while dsu[x] != x:
            dsu[x] = dsu[dsu[x]]
            x = dsu[x]
        return x
    
    def dsu_union(dsu, x, y):
        rx, ry = dsu_find(dsu, x), dsu_find(dsu, y)
        if rx == ry:
            return False
        dsu[ry] = rx
        return True
    
    def get_dsu(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in dsu_store:
            dsu_store[key] = list(range(n))
        return dsu_store[key]
    
    def can_assign(v, c):
        # Check adjacent vertices don't have color c
        for u in adj_list[v]:
            if colors[u] == c:
                return False
        
        # Check bicolored cycle: for each neighbor with different color,
        # check if v and u are already connected in the bicolored DSU
        for u in adj_list[v]:
            d = colors[u]
            if d >= 0 and d != c:
                dsu = get_dsu(c, d)
                if dsu_find(dsu, v) == dsu_find(dsu, u):
                    return False
        return True
    
    def assign(v, c):
        colors[v] = c
        for u in adj_list[v]:
            d = colors[u]
            if d >= 0 and d != c:
                dsu = get_dsu(c, d)
                dsu_union(dsu, v, u)
    
    for step in range(n):
        if not uncolored:
            break
        
        # Score each uncolored vertex: prefer those with fewest available colors
        best_v = None
        best_score = None
        best_c = None
        
        for v in uncolored:
            avail = []
            for c in range(k):
                if can_assign(v, c):
                    avail.append(c)
            if not avail:
                continue
            # Score: number of available colors (fewer = more constrained)
            score = len(avail)
            if best_score is None or score < best_score:
                best_score = score
                best_v = v
                best_c = avail[0]
        
        if best_v is None:
            return None
        
        # Assign the most constrained vertex
        assign(best_v, best_c)
        uncolored.remove(best_v)
    
    if -1 in colors:
        return None
    return colors

def _local_search(n, adj_list, colors, deadline):
    """Try to reduce the number of colors used."""
    if time.time() > deadline - 2:
        return colors
    
    k = len(set(colors))
    if k <= 3:
        return colors
    
    dsu_store = {}
    
    def dsu_find(dsu, x):
        while dsu[x] != x:
            dsu[x] = dsu[dsu[x]]
            x = dsu[x]
        return x
    
    def dsu_union(dsu, x, y):
        rx, ry = dsu_find(dsu, x), dsu_find(dsu, y)
        if rx == ry:
            return False
        dsu[ry] = rx
        return True
    
    def get_dsu(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in dsu_store:
            dsu_store[key] = list(range(n))
        return dsu_store[key]
    
    # Rebuild DSUs from current coloring
    for v in range(n):
        c = colors[v]
        for u in adj_list[v]:
            d = colors[u]
            if d != c:
                dsu = get_dsu(c, d)
                dsu_union(dsu, v, u)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try to remove one color
        for target_c in range(k - 1, -1, -1):
            # Find vertices with this color
            verts = [v for v in range(n) if colors[v] == target_c]
            if not verts:
                continue
            
            # Try to reassign each vertex to another color
            for v in verts:
                if time.time() > deadline - 2:
                    return colors
                for c in range(target_c):
                    # Check if we can change v from target_c to c
                    ok = True
                    for u in adj_list[v]:
                        if colors[u] == c:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # Check bicolored cycles for new assignments
                    # Remove v from target_c DSUs
                    for u in adj_list[v]:
                        d = colors[u]
                        if d != target_c:
                            pass  # We can't easily remove from DSU
                    
                    # Simple check: try the assignment and verify
                    old_c = colors[v]
                    colors[v] = c
                    
                    # Verify no bicolored cycle
                    valid = True
                    for u in adj_list[v]:
                        d = colors[u]
                        if d != c:
                            dsu = get_dsu(c, d)
                            # Check if v and u would be in same component
                            # We need a fresh check - this is tricky with shared DSUs
                            # For simplicity, do a BFS check
                            pass
                    
                    # Revert
                    colors[v] = old_c
                    
                    # Use a proper check
                    if _can_reassign(n, adj_list, colors, v, target_c, c):
                        colors[v] = c
                        # Update DSUs
                        for u in adj_list[v]:
                            d = colors[u]
                            if d != c:
                                dsu = get_dsu(c, d)
                                dsu_union(dsu, v, u)
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                k = len(set(colors))
                break
    
    return colors

def _can_reassign(n, adj_list, colors, v, old_c, new_c):
    """Check if vertex v can be changed from old_c to new_c without creating bicolored cycles."""
    # First check adjacency
    for u in adj_list[v]:
        if colors[u] == new_c:
            return False
    
    # For each neighbor with a color different from new_c, check if v and u
    # are already connected in the bicolored subgraph (excluding v's current edges)
    # We do a BFS in the bicolored subgraph induced by {new_c, d} 
    # starting from u, to see if we can reach v without using the edge (v,u)
    
    for u in adj_list[v]:
        d = colors[u]
        if d == new_c or d == old_c:
            continue  # d == old_c means the edge (v,u) is already in old_c-d subgraph
        
        # BFS in subgraph induced by colors {new_c, d}, starting from u
        # Can we reach v?
        visited = set()
        queue = [u]
        visited.add(u)
        while queue:
            cur = queue.pop(0)
            if cur == v:
                return False  # Would create a cycle
            for w in adj_list[cur]:
                if w in visited:
                    continue
                if colors[w] in (new_c, d):
                    visited.add(w)
                    queue.append(w)
    
    return True
