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

def _find_biclique_from_seed(seed_left, seed_right, edge_set, n, deadline):
    """Given a seed biclique, greedily expand it to be maximal."""
    if not seed_left or not seed_right:
        return None
    
    left = set(seed_left)
    right = set(seed_right)
    
    if left & right:
        # Remove overlap
        overlap = left & right
        left -= overlap
        right -= overlap
        if not left or not right:
            return None
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to add vertices to left
        for w in range(n):
            if w in left or w in right:
                continue
            if all(_edge_key((w, r)) in edge_set for r in right):
                left.add(w)
                changed = True
        # Try to add vertices to right
        for w in range(n):
            if w in left or w in right:
                continue
            if all(_edge_key((l, w)) in edge_set for l in left):
                right.add(w)
                changed = True
    
    return (left, right)

def _find_bicliques_greedy(remaining_edges, n, deadline):
    """Find bicliques greedily from remaining edges."""
    edge_set = set(_edge_key(e) for e in remaining_edges)
    bicliques = []
    rem = set(edge_set)
    
    while rem and time.time() < deadline:
        # Pick an edge to seed from
        # Use the edge with highest combined degree
        best_edge = None
        best_score = -1
        for e in list(rem):
            u, v = e
            score = len(edge_set)  # placeholder
            # Count how many edges share u or v
            shared = 0
            for e2 in rem:
                if u in e2 or v in e2:
                    shared += 1
            if shared > best_score:
                best_score = shared
                best_edge = e
        
        if best_edge is None:
            break
        
        u, v = best_edge
        # Try to find a biclique containing this edge
        # Start with {u} and {v}
        seed = _find_biclique_from_seed({u}, {v}, edge_set, n, deadline)
        if seed is None:
            # Just use the single edge
            seed = ({u}, {v})
        
        left, right = seed
        # Extract edges covered by this biclique
        covered = set()
        for a in left:
            for b in right:
                k = _edge_key((a, b))
                if k in edge_set:
                    covered.add(k)
        
        if covered:
            bicliques.append((left, right))
            rem -= covered
    
    return bicliques

def _try_merge_all(bicliques, all_edges_set, deadline):
    """Repeatedly try to merge any pair of bicliques."""
    bicliques = list(bicliques)
    merged = True
    while merged and time.time() < deadline:
        merged = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                return bicliques
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    return bicliques
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                # Try 4 combinations
                for combo in [(li | lj, ri | rj), (li | rj, ri | lj), (ri | lj, li | rj), (ri | rj, li | lj)]:
                    new_left, new_right = combo
                    if not new_left or not new_right:
                        continue
                    if new_left & new_right:
                        continue
                    
                    # Check all cross pairs are edges
                    valid = True
                    for u in new_left:
                        for v in new_right:
                            if _edge_key((u, v)) not in all_edges_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Check merged cross-edges = union of original cross-edges
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
            if merged:
                break
    return bicliques

def _try_split_and_repack(bicliques, all_edges_set, n, deadline):
    """Try to split bicliques and repack for better coverage."""
    # For now, just try merging
    return bicliques

def _validate_partition(bicliques, all_edges):
    """Check that bicliques form a valid partition of all edges."""
    all_edge_set = set(_edge_key(e) for e in all_edges)
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if left & right:
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

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    all_edges_set = set(_edge_key(e) for e in edges)
    
    # Strategy 1: Greedy biclique extraction
    bicliques = _find_bicliques_greedy(edges, n, deadline)
    
    # Try to merge
    bicliques = _try_merge_all(bicliques, all_edges_set, deadline)
    
    # Validate
    if not _validate_partition(bicliques, edges):
        # Fallback: one biclique per edge (always valid)
        result = []
        for u, v in edges:
            result.append([[u], [v]])
        return result
    
    # Convert to proper format
    result = []
    for left, right in bicliques:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
