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

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_biclique_from_seed(adj, remaining_edges, left_seed, right_seed, deadline):
    """Expand a biclique from seeds using iterative common-neighbor growth."""
    left_set = set(left_seed)
    right_set = set(right_seed)
    
    # Phase 1: Grow left by adding vertices connected to ALL right vertices
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find candidates that connect to all right vertices via remaining edges
        candidates = []
        for v in range(P['n']):
            if v in left_set or v in right_set:
                continue
            can_add = True
            for r in right_set:
                key = _edge_key(v, r)
                if key not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                candidates.append(v)
        if candidates:
            # Add all candidates at once (they all connect to all right)
            left_set.update(candidates)
            changed = True
    
    # Phase 2: Grow right by adding vertices connected to ALL left vertices
    changed = True
    while changed and time.time() < deadline:
        changed = False
        candidates = []
        for v in range(P['n']):
            if v in left_set or v in right_set:
                continue
            can_add = True
            for l in left_set:
                key = _edge_key(v, l)
                if key not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                candidates.append(v)
        if candidates:
            right_set.update(candidates)
            changed = True
    
    # Phase 3: Alternate growth to capture more
    for _ in range(5):
        if time.time() > deadline:
            break
        # Grow left
        added = False
        for v in range(P['n']):
            if v in left_set or v in right_set:
                continue
            can_add = True
            for r in right_set:
                if _edge_key(v, r) not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                left_set.add(v)
                added = True
        if added:
            # Grow right
            for v in range(P['n']):
                if v in left_set or v in right_set:
                    continue
                can_add = True
                for l in left_set:
                    if _edge_key(v, l) not in remaining_edges:
                        can_add = False
                        break
                if can_add:
                    right_set.add(v)
    
    if not left_set or not right_set:
        return None
    return (list(left_set), list(right_set))

def _find_best_biclique(adj, remaining_edges, deadline):
    """Find the largest biclique in the remaining edge set."""
    if not remaining_edges:
        return None
    
    # Compute remaining degrees
    deg = {}
    for (u, v) in remaining_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    
    # Sort vertices by remaining degree
    sorted_v = sorted(deg.keys(), key=lambda x: -deg[x])
    
    best = None
    best_size = 0
    
    # Try seeding from high-degree vertex pairs
    for i, a in enumerate(sorted_v[:15]):
        if time.time() > deadline:
            break
        for b in sorted_v[i+1:i+15]:
            if time.time() > deadline:
                break
            key = _edge_key(a, b)
            if key not in remaining_edges:
                continue
            # Seed with a on left, b on right
            bl = _find_biclique_from_seed(adj, remaining_edges, [a], [b], deadline)
            if bl:
                size = len(bl[0]) * len(bl[1])
                if size > best_size:
                    best_size = size
                    best = bl
    
    # Also try single-vertex seeds
    for a in sorted_v[:10]:
        if time.time() > deadline:
            break
        # Try a as left with each neighbor as right
        for b in list(adj[a])[:8]:
            key = _edge_key(a, b)
            if key not in remaining_edges:
                continue
            bl = _find_biclique_from_seed(adj, remaining_edges, [a], [b], deadline)
            if bl:
                size = len(bl[0]) * len(bl[1])
                if size > best_size:
                    best_size = size
                    best = bl
    
    if best is None:
        e = next(iter(remaining_edges))
        best = ([e[0]], [e[1]])
    
    return best

def _greedy_partition(adj, deadline):
    """Greedy: repeatedly find a large biclique among remaining edges."""
    remaining = set()
    for a, b in P['edges']:
        remaining.add(_edge_key(a, b))
    
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        
        bl = _find_best_biclique(adj, remaining, deadline)
        if bl is None:
            e = next(iter(remaining))
            bl = ([e[0]], [e[1]])
        
        l, r = bl
        bicliques.append((l, r))
        
        # Remove all cross edges from remaining
        for u in l:
            for v in r:
                remaining.discard(_edge_key(u, v))
    
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
                s_l1, s_r1, s_l2, s_r2 = set(l1), set(r1), set(l2), set(r2)
                
                # Try all 4 merge orientations
                merged = None
                for (new_l, new_r) in [
                    (s_l1 | s_l2, s_r1 & s_r2),
                    (s_l1 & s_l2, s_r1 | s_r2),
                    (s_l1 | s_r2, s_r1 & s_l2),
                    (s_l2 | s_r1, s_r2 & s_l1),
                ]:
                    if not new_l or not new_r:
                        continue
                    if new_l & new_r:
                        continue
                    # Verify all cross pairs are edges in the original graph
                    valid = True
                    for u in new_l:
                        for v in new_r:
                            if v not in adj[u]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Check that merged covers exactly the union of edges from i and j
                        edges_covered = set()
                        for u in new_l:
                            for v in new_r:
                                edges_covered.add(_edge_key(u, v))
                        old_edges = set()
                        for u in l1:
                            for v in r1:
                                old_edges.add(_edge_key(u, v))
                        for u in l2:
                            for v in r2:
                                old_edges.add(_edge_key(u, v))
                        if edges_covered == old_edges:
                            merged = (list(new_l), list(new_r))
                            break
                if merged:
                    bicliques[i] = merged
                    bicliques.pop(j)
                    changed = True
                    break
            if changed:
                break
    return bicliques

def _absorb_pass(bicliques, adj, deadline):
    """Try to absorb a smaller biclique's edges into a larger one by expanding it."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(n):
                if i == j:
                    continue
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                # Try to absorb j's edges into i by expanding i
                # New left = li ∪ lj, new right = ri ∪ rj (but must be valid biclique)
                # Actually, we need: all cross edges of (li∪lj, ri∪rj) to be in graph
                # AND the covered edges must equal union of edges from i and j
                
                new_l = set(li) | set(lj)
                new_r = set(ri) | set(rj)
                
                if new_l & new_r:
                    continue
                
                # Check if (new_l, new_r) is a valid biclique
                valid = True
                for u in new_l:
                    for v in new_r:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check edge coverage
                new_edges = set()
                for u in new_l:
                    for v in new_r:
                        new_edges.add(_edge_key(u, v))
                old_edges = set()
                for u in li:
                    for v in ri:
                        old_edges.add(_edge_key(u, v))
                for u in lj:
                    for v in rj:
                        old_edges.add(_edge_key(u, v))
                
                if new_edges == old_edges:
                    bicliques[i] = (list(new_l), list(new_r))
                    bicliques.pop(j)
                    changed = True
                    break
            if changed:
                break
    return bicliques

def _refine_pass(bicliques, adj, deadline):
    """Try to move vertices between bicliques to enable further merges."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                break
            li, ri = bicliques[i]
            # Try moving a vertex from left to right or vice versa
            for v in list(li):
                # Try moving v from left to right
                new_li = set(li) - {v}
                new_ri = set(ri) | {v}
                if not new_li or not new_ri:
                    continue
                if new_li & new_ri:
                    continue
                # Check validity
                valid = True
                for u in new_li:
                    for w in new_ri:
                        if w not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Check that covered edges are same
                    new_edges = set()
                    for u in new_li:
                        for w in new_ri:
                            new_edges.add(_edge_key(u, w))
                    old_edges = set()
                    for u in li:
                        for w in ri:
                            old_edges.add(_edge_key(u, w))
                    if new_edges == old_edges:
                        bicliques[i] = (list(new_li), list(new_ri))
                        changed = True
                        break
            if changed:
                break
        if changed:
            continue
        # Also try moving from right to left
        for i in range(n):
            if time.time() > deadline:
                break
            li, ri = bicliques[i]
            for v in list(ri):
                new_li = set(li) | {v}
                new_ri = set(ri) - {v}
                if not new_li or not new_ri:
                    continue
                if new_li & new_ri:
                    continue
                valid = True
                for u in new_li:
                    for w in new_ri:
                        if w not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_edges = set()
                    for u in new_li:
                        for w in new_ri:
                            new_edges.add(_edge_key(u, w))
                    old_edges = set()
                    for u in li:
                        for w in ri:
                            old_edges.add(_edge_key(u, w))
                    if new_edges == old_edges:
                        bicliques[i] = (list(new_li), list(new_ri))
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
    deadline = start_time + 85  # safety margin before 90s kill
    
    adj = _build_adj()
    
    # Run greedy with multiple random restarts
    best_bicliques = None
    best_count = float('inf')
    
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        random.seed(restart)
        bl = _greedy_partition(adj, deadline - 10)
        if bl:
            bl = _merge_pass(bl, adj, deadline - 5)
            bl = _absorb_pass(bl, adj, deadline - 3)
            bl = _refine_pass(bl, adj, deadline - 2)
            # After refine, try merge again
            bl = _merge_pass(bl, adj, deadline - 1)
            if len(bl) < best_count:
                best_count = len(bl)
                best_bicliques = bl
    
    if best_bicliques is None:
        # Fallback: star partition (always valid)
        best_bicliques = [[u, [v for a, v in P['edges'] if a == u]] 
                         for u in range(P['n']) if any(a == u for a, v in P['edges'])]
    
    return _to_result(best_bicliques)
# EVOLVE-BLOCK-END
