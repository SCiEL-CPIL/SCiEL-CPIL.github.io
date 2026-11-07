"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _find_max_biclique(adj, remaining_edges, anchor):
    """Find a large biclique anchored at `anchor` using only remaining edges."""
    # Candidates on the right: neighbors of anchor that still have an edge to anchor
    right_candidates = [v for v in adj[anchor] if (anchor, v) in remaining_edges or (v, anchor) in remaining_edges]
    if not right_candidates:
        return None
    
    # Greedily expand: start with right_candidates, iteratively add left vertices
    # that are connected to ALL current right vertices (via remaining edges)
    right_set = set(right_candidates)
    left_set = {anchor}
    
    # Iterative expansion
    changed = True
    while changed:
        changed = False
        # Try to add left vertices
        for v in range(P['n']):
            if v in left_set or v in right_set:
                continue
            # Check if v can be added to left: v must have remaining edges to all in right_set
            can_add = True
            for r in right_set:
                key = (min(v, r), max(v, r))
                if key not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                left_set.add(v)
                changed = True
        # Try to add right vertices
        for v in range(P['n']):
            if v in left_set or v in right_set:
                continue
            can_add = True
            for l in left_set:
                key = (min(v, l), max(v, l))
                if key not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                right_set.add(v)
                changed = True
    
    if not left_set or not right_set:
        return None
    return (list(left_set), list(right_set))

def _greedy_partition(adj, deadline):
    """Greedy: repeatedly find a large biclique among remaining edges."""
    remaining = set()
    for a, b in P['edges']:
        remaining.add((min(a, b), max(a, b)))
    
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        
        # Try multiple anchor strategies
        best_biclique = None
        best_size = 0
        
        # Strategy 1: anchor at vertex with most remaining edges
        deg_remaining = {}
        for (u, v) in remaining:
            deg_remaining[u] = deg_remaining.get(u, 0) + 1
            deg_remaining[v] = deg_remaining.get(v, 0) + 1
        
        # Sort anchors by remaining degree
        anchors = sorted(deg_remaining.keys(), key=lambda x: -deg_remaining[x])
        
        for anchor in anchors[:20]:
            bl = _find_max_biclique(adj, remaining, anchor)
            if bl:
                size = len(bl[0]) * len(bl[1])
                if size > best_size:
                    best_size = size
                    best_biclique = bl
        
        if best_biclique is None:
            # Fallback: just take one edge as a 1x1 biclique
            e = next(iter(remaining))
            best_biclique = ([e[0]], [e[1]])
        
        l, r = best_biclique
        bicliques.append((l, r))
        
        # Remove all cross edges from remaining
        for u in l:
            for v in r:
                key = (min(u, v), max(u, v))
                remaining.discard(key)
    
    return bicliques

def _merge_pass(bicliques, adj, deadline):
    """Try to merge pairs of bicliques if their union forms a valid biclique."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                l1, r1 = bicliques[i]
                l2, r2 = bicliques[j]
                # Try all 4 merge orientations:
                # (l1∪l2, r1∩r2), (l1∩l2, r1∪r2), (l1∪r2, r1∩l2), (l2∪r1, r2∩l1)
                merged = None
                for (new_l, new_r) in [
                    (set(l1)|set(l2), set(r1)&set(r2)),
                    (set(l1)&set(l2), set(r1)|set(r2)),
                    (set(l1)|set(r2), set(r1)&set(l2)),
                    (set(l2)|set(r1), set(r2)&set(l1)),
                ]:
                    if not new_l or not new_r:
                        continue
                    if new_l & new_r:
                        continue
                    # Verify all cross pairs are edges
                    valid = True
                    for u in new_l:
                        for v in new_r:
                            if v not in adj[u]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        merged = (list(new_l), list(new_r))
                        break
                if merged:
                    # Check that merged biclique covers exactly the union of edges from i and j
                    # (no extra edges beyond what was already covered)
                    edges_covered = set()
                    for u in merged[0]:
                        for v in merged[1]:
                            edges_covered.add((min(u,v), max(u,v)))
                    old_edges = set()
                    for u in l1:
                        for v in r1:
                            old_edges.add((min(u,v), max(u,v)))
                    for u in l2:
                        for v in r2:
                            old_edges.add((min(u,v), max(u,v)))
                    if edges_covered == old_edges:
                        bicliques[i] = merged
                        bicliques.pop(j)
                        changed = True
                        break
            if changed:
                break
    return bicliques

def _to_result(bicliques):
    """Convert to the required format with proper types."""
    result = []
    for l, r in bicliques:
        result.append([list(l), list(r)])
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    adj = _build_adj()
    
    # Run greedy with multiple random restarts
    best_bicliques = None
    best_count = float('inf')
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        random.seed(restart)
        bl = _greedy_partition(adj, deadline - 5)
        if bl:
            bl = _merge_pass(bl, adj, deadline - 2)
            if len(bl) < best_count:
                best_count = len(bl)
                best_bicliques = bl
    
    if best_bicliques is None:
        # Fallback: star partition (always valid)
        best_bicliques = [[u, [v for a, v in P['edges'] if a == u]] 
                         for u in range(P['n']) if any(a == u for a, v in P['edges'])]
    
    return _to_result(best_bicliques)
# EVOLVE-BLOCK-END
