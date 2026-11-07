import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_biclique_from_seed(uncovered_edges, adj, n, seed_u, seed_v, deadline):
    """Find a large biclique starting from edge (seed_u, seed_v) within uncovered edges."""
    # Build set of uncovered edges for fast lookup
    unc_set = set()
    for e in uncovered_edges:
        unc_set.add(_edge_key(*e))
    
    L = {seed_u}
    R = {seed_v}
    
    # Iteratively expand: try to add vertices that are connected to all on the other side
    changed = True
    while changed:
        changed = False
        
        # Try adding to L
        for w in range(n):
            if w in L or w in R:
                continue
            # w must be connected to all in R via uncovered edges
            can_add = True
            for r in R:
                if _edge_key(w, r) not in unc_set:
                    can_add = False
                    break
            if can_add:
                L.add(w)
                changed = True
        
        # Try adding to R
        for w in range(n):
            if w in L or w in R:
                continue
            can_add = True
            for l in L:
                if _edge_key(l, w) not in unc_set:
                    can_add = False
                    break
            if can_add:
                R.add(w)
                changed = True
        
        if time.time() > deadline:
            break
    
    return L, R

def _greedy_partition(edges, n, deadline, max_restarts=5):
    """Greedy biclique partition with multiple restarts using different edge orderings."""
    best_result = None
    best_count = float('inf')
    
    adj = _build_adj(n, edges)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Shuffle edges for diversity
        shuffled = list(edges)
        if restart > 0:
            random.shuffle(shuffled)
        
        uncovered = set()
        for u, v in shuffled:
            uncovered.add(_edge_key(u, v))
        
        bicliques = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick an edge to seed the biclique
            # Heuristic: pick edge with maximum degree in uncovered subgraph
            # For speed, just pick first few candidates and choose best expansion
            candidates = list(uncovered)[:min(10, len(uncovered))]
            
            best_biclique = None
            best_size = -1
            
            for cand in candidates:
                u, v = cand
                L, R = _find_max_biclique_from_seed(uncovered, adj, n, u, v, deadline)
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
            
            if best_biclique is None:
                # Fallback: single edge
                u, v = next(iter(uncovered))
                best_biclique = ({u}, {v})
            
            L, R = best_biclique
            
            # Remove all edges between L and R from uncovered
            for l in L:
                for r in R:
                    e = _edge_key(l, r)
                    if e in uncovered:
                        uncovered.discard(e)
            
            if L and R:
                bicliques.append([list(L), list(R)])
        
        # Cover any remaining uncovered edges with single-edge bicliques
        if uncovered:
            for e in uncovered:
                u, v = e
                bicliques.append([[u], [v]])
        
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_result = bicliques
    
    return best_result

def _try_merge(bicliques, edges_set, deadline):
    """Try to merge bicliques if their union is also a valid biclique."""
    if time.time() > deadline:
        return bicliques
    
    n = len(edges_set) if edges_set else 30
    # Actually, we need n from P
    n = P['n']
    
    adj = _build_adj(n, P['edges'])
    
    changed = True
    result = [list(b) for b in bicliques]
    
    while changed:
        changed = False
        for i in range(len(result)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(result)):
                if time.time() > deadline:
                    break
                L1, R1 = result[i]
                L2, R2 = result[j]
                
                # Try merging: new L = L1 ∪ L2, new R = R1 ∪ R2
                # Check if all cross pairs exist in original graph
                new_L = set(L1) | set(L2)
                new_R = set(R1) | set(R2)
                
                # Check disjointness
                if new_L & new_R:
                    continue
                
                # Check all cross edges exist
                valid = True
                for l in new_L:
                    for r in new_R:
                        if r not in adj[l]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    result[i] = [list(new_L), list(new_R)]
                    del result[j]
                    changed = True
                    break
            if changed:
                break
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Ensure deterministic behavior
    random.seed(42)
    
    # Run greedy partition with multiple restarts
    result = _greedy_partition(edges, n, deadline, max_restarts=3)
    
    if result is None:
        # Fallback: cover each edge individually
        result = [[[u], [v]] for u, v in edges]
    
    # Try to merge bicliques to reduce count
    if time.time() < deadline - 5.0:
        edges_set = set(_edge_key(u, v) for u, v in edges)
        result = _try_merge(result, edges_set, deadline)
    
    # Final validation and cleanup
    # Ensure all types are correct (Python int, list)
    final_result = []
    for biclique in result:
        left = [int(v) for v in biclique[0]]
        right = [int(v) for v in biclique[1]]
        if left and right:
            final_result.append([left, right])
    
    return final_result
