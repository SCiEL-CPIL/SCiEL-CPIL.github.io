"""Triangle-free diameter-four augmentation via hub-based construction."""
# EVOLVE-BLOCK-START

import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _build_adjacency(existing_edges, added_edges):
    """Build adjacency list from existing + added edges."""
    n = 24
    adj = {i: set() for i in range(n)}
    for u, v in existing_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_triangle_free(adj):
    """Check triangle-free by verifying bipartiteness (since graph is bipartite if triangle-free for this structure)."""
    # Actually check: no three vertices form a triangle
    n = 24
    for a in range(n):
        for b in adj[a]:
            if b > a:
                for c in adj[b]:
                    if c > b and c in adj[a]:
                        return False
    return True


def _diameter(adj):
    """Compute graph diameter via BFS from each vertex."""
    n = 24
    max_d = 0
    for start in range(n):
        dist = {start: 0}
        queue = [start]
        while queue:
            nxt = []
            for u in queue:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            queue = nxt
        if len(dist) < n:
            return -1  # not connected
        max_d = max(max_d, max(dist.values()))
    return max_d


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
    
    # Check triangle-free (bipartite check is sufficient for this structure, but do full check)
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
    1. Start with known good 10-edge solution
    2. Systematically try to remove edges (single and double removals)
    3. Try structured hub-based configurations with fewer edges
    4. Use fast validation
    """
    deadline = time.time() + 80  # 80s safety margin (limit is 90s)
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
    
    # Phase 1: Try removing single edges
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
    
    # Phase 3: Try structured configurations with fewer edges
    # Key insight: with diameter 4, we need efficient routing.
    # Try two-hub configurations where hubs are at different parities.
    
    even_verts = list(range(0, n, 2))  # 0,2,4,...,22
    odd_verts = list(range(1, n, 2))   # 1,3,5,...,23
    
    # For a hub h, it can directly connect to vertices of opposite parity.
    # A vertex v of same parity as h needs: v -> odd/even_neighbor -> h (2 steps)
    # or v -> ... -> h via path.
    
    # Try configurations with 8, 7, 6 edges
    for target_edges in range(best_val - 1, 3, -1):
        if time.time() > deadline:
            break
        
        # Try two-hub configurations
        for h1 in even_verts:
            if time.time() > deadline:
                break
            for h2 in odd_verts:
                if time.time() > deadline:
                    break
                
                # h1 (even) connects to some odd vertices
                # h2 (odd) connects to some even vertices
                # Plus edge (h1, h2)
                
                # Determine which vertices need connections:
                # For even vertex e: dist to h1 <= 2 means e is within 2 of h1 on path, 
                #   or e connects to an odd vertex that connects to h1
                # For odd vertex o: dist to h2 <= 2 means o is within 2 of h2 on path,
                #   or o connects to an even vertex that connects to h2
                
                # Simpler: try to find minimal connections
                # h1 needs to reach all even vertices within 2 steps
                # h2 needs to reach all odd vertices within 2 steps
                # Cross: even vertex to odd vertex should be within 4
                
                # Let's just try: h1 connects to odd vertices that are "far" from h1
                # and h2 connects to even vertices that are "far" from h2
                
                # Actually, let me try a specific pattern:
                # h1 = 6, h2 = 17 (or similar pairs)
                
                # For now, skip this and try another approach
                pass
        
        # Try: single hub with selective connections
        # Hub h (even): connects to odd vertices. 
        # Even vertex e: dist(e, h) on path. If dist > 2, need e to connect to an odd vertex 
        #   that is adjacent to h (added edge e->odd, and odd->h already exists or added).
        # Actually e is even, h is even. e->h requires e->odd_x->h where odd_x adj to both.
        # odd_x adj to h means h-odd_x is an edge (path or added).
        # odd_x adj to e means e-odd_x is an edge (path or added).
        
        # For hub 12: path neighbors are 11, 13 (both odd). 
        # Even vertex e needs an odd neighbor that is also neighbor of 12.
        # Odd neighbors of 12: 11, 13 (path) + added edges from 12 to odd.
        # If 12 connects to odd o, then even e that is path-adjacent to o can reach 12 in 2 steps.
        
        # This is complex. Let me try specific small configurations.
        
        # Configuration: hub 11 (odd) and hub 12 (even) - adjacent hubs
        # 11 connects to even: 10,12(path), and some others
        # 12 connects to odd: 11,13(path), and some others
        # This might work with fewer edges
        
        # Let me try: 8-edge configurations with two adjacent hubs
        # Hub A=11 (odd), Hub B=12 (even), edge (11,12) is in path
        
        # 11 (odd) added edges to even: need to cover even vertices far from 11
        # 12 (even) added edges to odd: need to cover odd vertices far from 12
        
        # Even vertices: 0,2,4,6,8,10,12,14,16,18,20,22
        # dist from 11 on path: 11,9,7,5,3,1,1,3,5,7,9,11
        # Even vertices with dist > 2 from 11: 0(11),2(9),4(7),6(5),14(3),16(5),18(7),20(9),22(11)
        # But they can also reach via 12: even -> 12's odd neighbor -> 12 -> 11 -> ... 
        # Actually even e to odd o (added to 12): e->o is 1 step if path-adjacent, 
        #   then o->12 is 1 step (added), then 12->11 is 1 step (path). Total 3.
        # So even e that is path-adjacent to an odd o connected to 12 can reach 11 in 3 steps.
        
        # This is getting very complex. Let me try a different approach:
        # Use the fast validator to test specific small configurations.
        
        # Try 8-edge configuration: hubs at 7 and 16
        # 7 (odd) connects to even: 4, 10, 14, 18, 22 (5 edges)
        # 16 (even) connects to odd: 3, 9, 13, 19, 23 (5 edges)  
        # Plus edge (7,16): 1 edge. Total: 11. Too many.
        
        # Try: hubs at 8 and 15
        # 8 (even) connects to odd: 5, 11, 13, 19, 23 (5 edges)
        # 15 (odd) connects to even: 4, 10, 12, 18, 22 (5 edges)
        # Plus edge (8,15): 1 edge. Total: 11.
        
        # I think 10 might be close to optimal. Let me verify the 10-edge solution
        # and try to find if 9 works with a smarter search.
        
        # Try all 9-edge subsets of the 10-edge solution's "neighborhood"
        # The 10-edge solution has hub 12. Let's see if we can replace some edges.
        
        # Alternative: try hub 11 instead of 12
        # Hub 11 (odd) connects to even vertices
        # Even vertices far from 11: 0,2,4,6,14,16,18,20,22
        # But 11 is adjacent to 10,12 on path.
        # If 11 connects to even e, then odd o path-adjacent to e can reach 11 in 2 steps.
        
        # Let me just try: hub 11 connects to {0,2,4,6,14,16,18,20,22} = 9 edges
        # Check if this gives diameter 4
        config_11 = [[11, e] for e in [0, 2, 4, 6, 14, 16, 18, 20, 22]]
        if len(config_11) < best_val and fast_validate(config_11):
            best = config_11
            best_val = len(best)
        
        # Hub 12 connects to {1,3,5,7,9,15,17,19,21,23} = 10 edges (original)
        
        # Try hub 11 with fewer: maybe some even vertices don't need direct connection
        # Even vertex 10 is adjacent to 11 on path, so dist(10,11)=1. Good.
        # Even vertex 12 is adjacent to 11 on path, so dist(12,11)=1. Good.
        # Even vertex 8: dist(8,11)=3 on path. Need shortcut.
        #   8 -> 9 -> 11? 9 is odd, adj to 8 and 10 on path. Not adj to 11.
        #   8 -> 7 -> 11? 7 adj to 8,6. Not 11.
        #   So 8 needs: 8->odd_x->11 where odd_x adj to both 8 and 11.
        #   odd_x adj to 11: 10,12 (path, but they're even!) or added.
        #   Wait, 11 is odd, its path neighbors are 10,12 (even).
        #   So for 8 to reach 11: 8->9->?->11 or 8->7->?->11
        #   9 is odd. 9 adj to 8,10. To reach 11 from 9: 9->10->11 (2 more steps). Total 3.
        #   So dist(8,11) = 3 via 8->9->10->11. That's fine!
        #   Similarly 6: 6->7->8->9->10->11 = 5 steps. Too far.
        #   6 needs shortcut: 6->odd_x->11 where odd_x adj to 6 and 11.
        #   odd_x adj to 6: 5,7. odd_x adj to 11: 10,12 (even, can't be odd_x) or added.
        #   So need added edge: either 6->10 (even-even, INVALID) or 6->12 (even-even, INVALID).
        #   Or: 6->5->11? 5 adj to 6,4. Not 11. 
        #   6->7->11? 7 adj to 6,8. Not 11.
        #   So 6 MUST have an added edge to an odd vertex that is also adjacent to 11.
        #   Odd vertices adjacent to 11: only 10,12 are even... 
        #   Wait, I'm confusing myself. 11 is odd. Its neighbors in the final graph are:
        #   Path: 10, 12 (both even)
        #   Added: even vertices we connect 11 to.
        #   So 11's neighbors are all even. 
        #   For 6 (even) to reach 11 (odd): 6->even? No, 6 is even, must go to odd.
        #   6->odd_x->11: odd_x must be adj to 6 and to 11.
        #   odd_x adj to 11: 11's odd neighbors? 11 is odd, its neighbors are even (10,12 + added evens).
        #   So 11 has NO odd neighbors! Therefore no odd_x is adjacent to 11.
        #   This means even vertex 6 CANNOT reach 11 in 2 steps via an odd intermediate!
        #   It must go: 6->odd_x->even_y->11 (3 steps) where odd_x adj to 6 and even_y, and even_y adj to 11.
        #   even_y adj to 11: even_y in {10, 12} (path) or added.
        #   odd_x adj to 6 and even_y: 
        #     If even_y=10: odd_x adj to 6 and 10. odd_x in {5,7} (adj to 6). 5 adj to 10? No (dist 5). 7 adj to 10? No (dist 3).
        #     If even_y=12: odd_x adj to 6 and 12. 5 adj to 12? No. 7 adj to 12? No.
        #     If even_y is added (11->even_y): odd_x adj to 6 and even_y.
        #       If even_y=8: 7 adj to 8? Yes! So 6->7->8->11 if 11->8 is added. 3 steps.
        #       If even_y=4: 5 adj to 4? Yes! So 6->5->4->11 if 11->4 is added. 3 steps.
        
        # This is very complex. The 10-edge hub-12 solution works because 12 is even and
        # connects to odd vertices, creating a bipartite structure where even vertices
        # can reach 12 via: even -> odd(path neighbor) -> 12(added) = 2 steps, or
        # even -> 12(path) = 1 step.
        
        # I'll stick with the 10-edge solution and try to verify it's minimal
        # by checking if any 9-edge solution exists via exhaustive search of "nearby" configs.
        
        # Try: remove one edge from hub-12 solution and add one different edge
        if time.time() < deadline:
            base_config = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                          [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
            for i in range(10):
                if time.time() > deadline:
                    break
                # Remove edge i
                remaining = base_config[:i] + base_config[i+1:]
                # Try adding one alternative edge
                for alt_u in even_verts:
                    if time.time() > deadline:
                        break
                    for alt_v in odd_verts:
                        if time.time() > deadline:
                            break
                        key = (min(alt_u, alt_v), max(alt_u, alt_v))
                        if key in base_edge_set:
                            continue
                        # Check not already in remaining
                        already = False
                        for ru, rv in remaining:
                            if (min(ru,rv), max(ru,rv)) == key:
                                already = True
                                break
                        if already:
                            continue
                        trial = remaining + [[alt_u, alt_v]]
                        if len(trial) < best_val and fast_validate(trial):
                            best = trial
                            best_val = len(best)
                            break
                    if len(best) < best_val:
                        break
    
    # Final validation and cleanup
    if not fast_validate(best):
        best = [list(e) for e in fallback]
    
    result = []
    for edge in best:
        result.append([int(edge[0]), int(edge[1])])
    
    return result

# EVOLVE-BLOCK-END
