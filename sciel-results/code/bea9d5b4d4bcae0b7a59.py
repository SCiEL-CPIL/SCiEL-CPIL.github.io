"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(e):
    a, b = e
    return (min(a, b), max(a, b))

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_bitsets(n, edges):
    """Build adjacency bitsets for fast intersection."""
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    return adj_bits

def _popcount(x):
    return bin(x).count('1')

def _find_biclique_bitset(adj_bits, n, remaining_mask, seed_vertex=None):
    """Find a large biclique using bitset operations.
    remaining_mask: for each vertex, which neighbors are still 'active' (have remaining edges to them)
    Returns (left_set, right_set) or None.
    """
    # Try to find a biclique by starting from a seed vertex
    if seed_vertex is None:
        # Pick vertex with highest remaining degree
        best_v = -1
        best_deg = 0
        for v in range(n):
            d = _popcount(remaining_mask[v])
            if d > best_deg:
                best_deg = d
                best_v = v
        if best_deg == 0:
            return None
        seed_vertex = best_v
    
    # Start with left={seed}, right = neighbors of seed in remaining
    left = {seed_vertex}
    right_bits = remaining_mask[seed_vertex]
    
    # Expand left: add vertices that connect to all of right
    changed = True
    while changed:
        changed = False
        for w in range(n):
            if w in left:
                continue
            # w must connect to all vertices in right
            # Check if remaining_mask[w] contains all of right_bits
            if right_bits and (remaining_mask[w] & right_bits) == right_bits:
                left.add(w)
                changed = True
    
    # Expand right: add vertices that connect to all of left
    changed = True
    while changed:
        changed = False
        # Compute intersection of remaining_mask for all left vertices
        common = (1 << n) - 1
        for u in left:
            common &= remaining_mask[u]
        # Remove vertices already in left
        for u in left:
            common &= ~(1 << u)
        # Add all vertices in common to right
        new_right = common
        for w in range(n):
            if (new_right >> w) & 1 and w not in left:
                if w not in set(bin(right_bits).replace('0b','').zfill(n)[::-1] if right_bits else []):
                    pass
        # Simpler: just set right_bits to common
        right_bits = common
        if _popcount(right_bits) > 0:
            # Check if we can add more to left
            pass
        changed = False  # We'll do one more pass below
    
    # Convert right_bits to set
    right = set()
    for w in range(n):
        if (right_bits >> w) & 1:
            right.add(w)
    
    # Remove any overlap
    right -= left
    
    if not left or not right:
        return None
    
    return (left, right)

def _find_max_biclique(remaining_edges, adj, n, deadline, adj_bits, remaining_masks):
    """Find a large biclique within remaining edges using multiple strategies."""
    if not remaining_edges:
        return None
    
    edge_set = set(_edge_key(e) for e in remaining_edges)
    
    # Build remaining adjacency masks
    rem_adj = [0] * n
    for u, v in remaining_edges:
        rem_adj[u] |= (1 << v)
        rem_adj[v] |= (1 << u)
    
    # Compute degrees
    deg = [_popcount(rem_adj[v]) for v in range(n)]
    
    # Strategy 1: Try from each high-degree vertex
    best_biclique = None
    best_size = 0
    
    verts_sorted = sorted(range(n), key=lambda x: -deg[x])
    top_k = min(20, len(verts_sorted))
    
    for i in range(top_k):
        if time.time() > deadline:
            break
        u = verts_sorted[i]
        if deg[u] < 1:
            continue
        
        # Try biclique with u on left
        left = {u}
        right_bits = rem_adj[u]
        
        # Expand left greedily
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in left:
                    continue
                if right_bits and (rem_adj[w] & right_bits) == right_bits:
                    left.add(w)
                    changed = True
        
        # Expand right greedily
        changed = True
        while changed:
            changed = False
            common = (1 << n) - 1
            for u2 in left:
                common &= rem_adj[u2]
            for u2 in left:
                common &= ~(1 << u2)
            # Check if any new vertices
            new_right_bits = common
            if new_right_bits != right_bits:
                right_bits = new_right_bits
                changed = True
        
        right = set()
        for w in range(n):
            if (right_bits >> w) & 1:
                right.add(w)
        right -= left
        
        if not right:
            continue
        
        # Verify all cross edges exist
        valid = True
        for u2 in left:
            for v2 in right:
                if _edge_key((u2, v2)) not in edge_set:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            size = len(left) * len(right)
            if size > best_size:
                best_size = size
                best_biclique = (left, right)
    
    # Strategy 2: Try from edges
    if best_biclique is None or best_size < 4:
        for u, v in remaining_edges[:30]:
            if time.time() > deadline:
                break
            left = {u}
            right = {v}
            
            # Expand
            changed = True
            while changed:
                changed = False
                # Expand left
                for w in range(n):
                    if w in left or w in right:
                        continue
                    if all(_edge_key((w, r)) in edge_set for r in right):
                        left.add(w)
                        changed = True
                # Expand right
                for w in range(n):
                    if w in left or w in right:
                        continue
                    if all(_edge_key((l, w)) in edge_set for l in left):
                        right.add(w)
                        changed = True
            
            size = len(left) * len(right)
            if size > best_size:
                best_size = size
                best_biclique = (left, right)
    
    if best_biclique is None:
        u, v = remaining_edges[0]
        return ({u}, {v})
    
    return best_biclique

def _extract_edges_from_biclique(biclique, edge_set):
    """Get the set of edges in edge_set that are cross-edges of the biclique."""
    left, right = biclique
    edges = set()
    for u in left:
        for v in right:
            k = _edge_key((u, v))
            if k in edge_set:
                edges.add(k)
    return edges

def _validate_partition(bicliques, all_edges):
    """Check that bicliques form a valid partition of all edges."""
    all_edge_set = set(_edge_key(e) for e in all_edges)
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left) & set(right):
            return False
        for u in left:
            for v in right:
                k = _edge_key((u, v))
                if k not in all_edge_set:
                    return False
                if k in covered:
                    return False
                covered.add(k)
    return covered == all_edge_set

def _try_merge(bicliques, all_edges, deadline):
    """Try to merge pairs of bicliques if compatible."""
    merged = True
    while merged and time.time() < deadline:
        merged = False
        bicliques = list(bicliques)
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                for combo in [(li | lj, ri | rj), (li | rj, ri | lj), (ri | lj, li | rj), (ri | rj, li | lj)]:
                    new_left, new_right = combo
                    if not new_left or not new_right:
                        continue
                    if new_left & new_right:
                        continue
                    
                    valid = True
                    for u in new_left:
                        for v in new_right:
                            k = _edge_key((u, v))
                            if k not in all_edges:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    cross_merged = set()
                    for u in new_left:
                        for v in new_right:
                            cross_merged.add(_edge_key((u, v)))
                    
                    cross_i = set()
                    for u in li:
                        for v in ri:
                            cross_i.add(_edge_key((u, v)))
                    cross_j = set()
                    for u in lj:
                        for v in rj:
                            cross_j.add(_edge_key((u, v)))
                    
                    if cross_merged == cross_i | cross_j:
                        bicliques[i] = (new_left, new_right)
                        bicliques.pop(j)
                        merged = True
                        break
            if merged:
                break
    return bicliques

def _local_search(bicliques, all_edges, n, deadline):
    """Local search: try to move vertices between bicliques or split/recombine."""
    bicliques = [list(bc) for bc in bicliques]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            li, ri = bicliques[i]
            # Try moving a vertex from left to right or vice versa
            for v in list(li):
                new_li = set(li) - {v}
                new_ri = set(ri) | {v}
                if not new_li or not new_ri or new_li & new_ri:
                    continue
                # Check validity
                valid = True
                for u in new_li:
                    for w in new_ri:
                        if _edge_key((u, w)) not in all_edges:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Check that cross edges still partition correctly
                    # (simplified: just accept if valid)
                    bicliques[i] = (new_li, new_ri)
                    improved = True
            for v in list(ri):
                new_li = set(li) | {v}
                new_ri = set(ri) - {v}
                if not new_li or not new_ri or new_li & new_ri:
                    continue
                valid = True
                for u in new_li:
                    for w in new_ri:
                        if _edge_key((u, w)) not in all_edges:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    bicliques[i] = (new_li, new_ri)
                    improved = True
    
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    all_edges = set(_edge_key(e) for e in edges)
    adj = _build_adj(n, edges)
    adj_bits = _build_bitsets(n, edges)
    
    # Greedy extraction with multiple restarts
    best_bicliques = None
    best_count = float('inf')
    
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        
        remaining = set(all_edges)
        bicliques = []
        
        while remaining and time.time() < deadline - 5:
            rem_list = [(u, v) for u, v in remaining]
            
            biclique = _find_max_biclique(rem_list, adj, n, deadline - 5, adj_bits, None)
            if biclique is None:
                break
            
            left, right = biclique
            cross_edges = _extract_edges_from_biclique(biclique, remaining)
            
            if not cross_edges:
                break
            
            bicliques.append((left, right))
            remaining -= cross_edges
        
        for u, v in remaining:
            bicliques.append(({u}, {v}))
        
        # Merge
        bicliques = _try_merge(bicliques, all_edges, deadline - 5)
        
        # Local search
        bicliques = _local_search(bicliques, all_edges, n, deadline - 5)
        
        # Validate
        if _validate_partition(bicliques, edges):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_bicliques = bicliques
    
    if best_bicliques is None:
        # Fallback
        return [[[u], [v for a, v in edges if a == u]] for u in range(n) if any(a == u for a, v in edges)]
    
    # Convert to proper format
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
