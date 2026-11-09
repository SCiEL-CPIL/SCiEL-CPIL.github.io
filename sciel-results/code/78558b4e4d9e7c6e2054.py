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
            return -1
        max_d = max(max_d, max(dist.values()))
    return max_d


def _validate(added_edges):
    """Validate the added edges: triangle-free, connected, diameter <= 4, no duplicates, no existing edges."""
    n = 24
    base_edge_set = set()
    for u, v in P['edges']:
        base_edge_set.add((min(u, v), max(u, v)))
    
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
    
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for a in range(n):
        for b in adj[a]:
            if b > a:
                for c in adj[b]:
                    if c > b and c in adj[a]:
                        return False
    
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
    2. Systematically try to remove edges
    3. Try structured two-hub and multi-hub configurations
    4. Use fast validation with bitmasks
    """
    deadline = time.time() + 85  # 85s safety margin (limit is 90s)
    n = 24
    
    # Known good solution: hub 12 connects to all odd vertices except 11,13
    fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    
    best = [list(e) for e in fallback]
    best_val = len(best)
    
    # Precompute base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
    base_edge_set = set((min(u,v), max(u,v)) for u, v in P['edges'])
    
    def fast_validate(added_edges):
        """Fast validation: triangle-free, connected, diameter <= 4."""
        seen = set()
        for u, v in added_edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u,v), max(u,v))
            if key in base_edge_set or key in seen:
                return False
            seen.add(key)
        
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
    
    # Phase 1: Try removing single edges from current best
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
    
    # Phase 3: Systematic two-hub search
    # Key insight: With two hubs h1 (even) and h2 (odd), we need:
    # - h1 connects to some odd vertices, h2 connects to some even vertices
    # - All vertices must be within distance 4 of all others
    # 
    # For a vertex v to be within dist 4 of all others via hubs:
    # - If v is even: v can reach h1 in at most 2 steps (v->odd->h1) if v's path-neighbor odd is connected to h1, or v->h1 directly if path-adjacent
    # - If v is odd: v can reach h2 in at most 2 steps similarly
    # - Cross parity: even e to odd o: e->h1->h2->o or e->h1->...->o
    #
    # Optimal strategy: choose h1, h2 to minimize total edges needed
    
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    # For each pair of hubs, compute minimum edges needed
    for h1 in even_verts:
        if time.time() > deadline:
            break
        for h2 in odd_verts:
            if time.time() > deadline:
                break
            
            # h1 (even) needs to connect to odd vertices so that all even vertices
            # can reach h1 within 2 steps
            # Even vertex e reaches h1 in 2 steps if: e is path-adjacent to an odd o 
            # that is connected to h1 (via added edge or path)
            # OR e is path-adjacent to h1 (dist 1)
            # OR e = h1 (dist 0)
            
            # Odd vertices that are path-adjacent to h1: h1-1, h1+1 (if valid)
            # These are already "covered" for h1 without added edges
            
            # For even vertex e != h1, e needs an odd neighbor o such that o is 
            # connected to h1. o is path-adjacent to e (o = e-1 or e+1).
            # o is connected to h1 if: o is path-adjacent to h1 OR (h1,o) is an added edge.
            
            # So for each even e, we need at least one of {e-1, e+1} to be connected to h1.
            # An odd o is "connected to h1" if o in {h1-1, h1+1} or (h1,o) in added_edges.
            
            # Similarly for h2 (odd): for each odd o != h2, need at least one of 
            # {o-1, o+1} to be connected to h2.
            
            # Let's compute which odd vertices need to be explicitly connected to h1
            # (i.e., not already path-adjacent to h1)
            
            # Odd vertices path-adjacent to h1:
            h1_odd_neighbors = set()
            if h1 - 1 >= 0:
                h1_odd_neighbors.add(h1 - 1)
            if h1 + 1 < n:
                h1_odd_neighbors.add(h1 + 1)
            
            # For each even vertex e (except h1), find which odd neighbors it has
            # and whether any are already connected to h1
            h1_needed_odds = set()
            for e in even_verts:
                if e == h1:
                    continue
                # e's odd neighbors: e-1, e+1
                e_odd_neighbors = set()
                if e - 1 >= 0:
                    e_odd_neighbors.add(e - 1)
                if e + 1 < n:
                    e_odd_neighbors.add(e + 1)
                
                # Check if any of e's odd neighbors are already connected to h1
                covered = False
                for o in e_odd_neighbors:
                    if o in h1_odd_neighbors:
                        covered = True
                        break
                if not covered:
                    # Need to add at least one of e's odd neighbors to h1
                    # We'll add the one that also helps other even vertices
                    h1_needed_odds.update(e_odd_neighbors)
            
            # Similarly for h2 (odd): which even vertices need to be connected to h2?
            h2_even_neighbors = set()
            if h2 - 1 >= 0:
                h2_even_neighbors.add(h2 - 1)
            if h2 + 1 < n:
                h2_even_neighbors.add(h2 + 1)
            
            h2_needed_evens = set()
            for o in odd_verts:
                if o == h2:
                    continue
                o_even_neighbors = set()
                if o - 1 >= 0:
                    o_even_neighbors.add(o - 1)
                if o + 1 < n:
                    o_even_neighbors.add(o + 1)
                
                covered = False
                for e in o_even_neighbors:
                    if e in h2_even_neighbors:
                        covered = True
                        break
                if not covered:
                    h2_needed_evens.update(o_even_neighbors)
            
            # Now we need to select a subset of h1_needed_odds to connect to h1
            # such that all even vertices are covered, and similarly for h2.
            # This is a set cover problem, but small enough to try greedily.
            
            # Greedy: for h1, for each even vertex not yet covered, pick the odd
            # neighbor that covers the most uncovered even vertices.
            
            def min_set_cover(elements_to_cover, candidate_sets):
                """Greedy set cover. elements_to_cover: set of elements. 
                candidate_sets: list of (set, cost) where cost is always 1 here."""
                uncovered = set(elements_to_cover)
                selected = []
                while uncovered:
                    best_set = None
                    best_count = -1
                    for s in candidate_sets:
                        count = len(uncovered & s)
                        if count > best_count:
                            best_count = count
                            best_set = s
                    if best_set is None or best_count == 0:
                        return None  # Can't cover all
                    selected.append(best_set)
                    uncovered -= best_set
                return selected
            
            # For h1: candidate sets are the odd neighbors of each even vertex
            # We want to select odd vertices to connect to h1
            # Each odd o "covers" the even vertices that have o as a neighbor
            
            # Build: for each odd o in h1_needed_odds, which even vertices does it cover?
            odd_to_evens = {}
            for o in h1_needed_odds:
                evens_covered = set()
                for e in even_verts:
                    if e == h1:
                        continue
                    if e - 1 == o or e + 1 == o:
                        evens_covered.add(e)
                odd_to_evens[o] = evens_covered
            
            # Even vertices that need coverage (those not already covered by h1_odd_neighbors)
            evens_needing_h1 = set()
            for e in even_verts:
                if e == h1:
                    continue
                e_odd_neighbors = set()
                if e - 1 >= 0:
                    e_odd_neighbors.add(e - 1)
                if e + 1 < n:
                    e_odd_neighbors.add(e + 1)
                if not (e_odd_neighbors & h1_odd_neighbors):
                    evens_needing_h1.add(e)
            
            # Greedy set cover for h1
            h1_selected_odds = []
            uncovered_evens = set(evens_needing_h1)
            while uncovered_evens:
                best_o = None
                best_count = -1
                for o in h1_needed_odds:
                    if o in [s for s in h1_selected_odds]:
                        continue
                    count = len(uncovered_evens & odd_to_evens.get(o, set()))
                    if count > best_count:
                        best_count = count
                        best_o = o
                if best_o is None or best_count == 0:
                    break  # Can't cover all
                h1_selected_odds.append(best_o)
                uncovered_evens -= odd_to_evens.get(best_o, set())
            
            if uncovered_evens:
                continue  # Can't cover all even vertices with h1
            
            # Similarly for h2
            even_to_odds = {}
            for e in h2_needed_evens:
                odds_covered = set()
                for o in odd_verts:
                    if o == h2:
                        continue
                    if o - 1 == e or o + 1 == e:
                        odds_covered.add(o)
                even_to_odds[e] = odds_covered
            
            odds_needing_h2 = set()
            for o in odd_verts:
                if o == h2:
                    continue
                o_even_neighbors = set()
                if o - 1 >= 0:
                    o_even_neighbors.add(o - 1)
                if o + 1 < n:
                    o_even_neighbors.add(o + 1)
                if not (o_even_neighbors & h2_even_neighbors):
                    odds_needing_h2.add(o)
            
            h2_selected_evens = []
            uncovered_odds = set(odds_needing_h2)
            while uncovered_odds:
                best_e = None
                best_count = -1
                for e in h2_needed_evens:
                    if e in [s for s in h2_selected_evens]:
                        continue
                    count = len(uncovered_odds & even_to_odds.get(e, set()))
                    if count > best_count:
                        best_count = count
                        best_e = e
                if best_e is None or best_count == 0:
                    break
                h2_selected_evens.append(best_e)
                uncovered_odds -= even_to_odds.get(best_e, set())
            
            if uncovered_odds:
                continue  # Can't cover all odd vertices with h2
            
            # Total edges: len(h1_selected_odds) + len(h2_selected_evens)
            # Plus we need (h1, h2) edge if not already in path
            total_edges = len(h1_selected_odds) + len(h2_selected_evens)
            if (min(h1, h2), max(h1, h2)) not in base_edge_set:
                total_edges += 1
            
            if total_edges >= best_val:
                continue
            
            # Build the candidate configuration
            config = []
            for o in h1_selected_odds:
                config.append([h1, o])
            for e in h2_selected_evens:
                config.append([h2, e])
            if (min(h1, h2), max(h1, h2)) not in base_edge_set:
                config.append([h1, h2])
            
            if len(config) < best_val and fast_validate(config):
                best = config
                best_val = len(best)
    
    # Phase 4: Try single hub configurations with different hubs
    for h in even_verts:
        if time.time() > deadline:
            break
        # h (even) connects to odd vertices
        # Need all even vertices to reach h within 2 steps
        h_odd_neighbors = set()
        if h - 1 >= 0:
            h_odd_neighbors.add(h - 1)
        if h + 1 < n:
            h_odd_neighbors.add(h + 1)
        
        # For each even e != h, need an odd neighbor connected to h
        needed_odds = set()
        for e in even_verts:
            if e == h:
                continue
            e_odd_neighbors = set()
            if e - 1 >= 0:
                e_odd_neighbors.add(e - 1)
            if e + 1 < n:
                e_odd_neighbors.add(e + 1)
            if not (e_odd_neighbors & h_odd_neighbors):
                needed_odds.update(e_odd_neighbors)
        
        # Greedy set cover
        odd_to_evens = {}
        for o in needed_odds:
            evens_covered = set()
            for e in even_verts:
                if e == h:
                    continue
                if e - 1 == o or e + 1 == o:
                    evens_covered.add(e)
            odd_to_evens[o] = evens_covered
        
        evens_needing = set()
        for e in even_verts:
            if e == h:
                continue
            e_odd_neighbors = set()
            if e - 1 >= 0:
                e_odd_neighbors.add(e - 1)
            if e + 1 < n:
                e_odd_neighbors.add(e + 1)
            if not (e_odd_neighbors & h_odd_neighbors):
                evens_needing.add(e)
        
        selected_odds = []
        uncovered = set(evens_needing)
        while uncovered:
            best_o = None
            best_count = -1
            for o in needed_odds:
                if o in selected_odds:
                    continue
                count = len(uncovered & odd_to_evens.get(o, set()))
                if count > best_count:
                    best_count = count
                    best_o = o
            if best_o is None or best_count == 0:
                break
            selected_odds.append(best_o)
            uncovered -= odd_to_evens.get(best_o, set())
        
        if uncovered:
            continue
        
        config = [[h, o] for o in selected_odds]
        if len(config) < best_val and fast_validate(config):
            best = config
            best_val = len(best)
    
    for h in odd_verts:
        if time.time() > deadline:
            break
        h_even_neighbors = set()
        if h - 1 >= 0:
            h_even_neighbors.add(h - 1)
        if h + 1 < n:
            h_even_neighbors.add(h + 1)
        
        needed_evens = set()
        for o in odd_verts:
            if o == h:
                continue
            o_even_neighbors = set()
            if o - 1 >= 0:
                o_even_neighbors.add(o - 1)
            if o + 1 < n:
                o_even_neighbors.add(o + 1)
            if not (o_even_neighbors & h_even_neighbors):
                needed_evens.update(o_even_neighbors)
        
        even_to_odds = {}
        for e in needed_evens:
            odds_covered = set()
            for o in odd_verts:
                if o == h:
                    continue
                if o - 1 == e or o + 1 == e:
                    odds_covered.add(o)
            even_to_odds[e] = odds_covered
        
        odds_needing = set()
        for o in odd_verts:
            if o == h:
                continue
            o_even_neighbors = set()
            if o - 1 >= 0:
                o_even_neighbors.add(o - 1)
            if o + 1 < n:
                o_even_neighbors.add(o + 1)
            if not (o_even_neighbors & h_even_neighbors):
                odds_needing.add(o)
        
        selected_evens = []
        uncovered = set(odds_needing)
        while uncovered:
            best_e = None
            best_count = -1
            for e in needed_evens:
                if e in selected_evens:
                    continue
                count = len(uncovered & even_to_odds.get(e, set()))
                if count > best_count:
                    best_count = count
                    best_e = e
            if best_e is None or best_count == 0:
                break
            selected_evens.append(best_e)
            uncovered -= even_to_odds.get(best_e, set())
        
        if uncovered:
            continue
        
        config = [[h, e] for e in selected_evens]
        if len(config) < best_val and fast_validate(config):
            best = config
            best_val = len(best)
    
    # Final validation and cleanup
    if not fast_validate(best):
        best = [list(e) for e in fallback]
    
    result = []
    for edge in best:
        result.append([int(edge[0]), int(edge[1])])
    
    return result

# EVOLVE-BLOCK-END
