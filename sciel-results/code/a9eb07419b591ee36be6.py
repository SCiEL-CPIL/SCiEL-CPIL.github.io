import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _maximal_biclique_from_anchor(adj, anchor, remaining, n, deadline):
    """Build a maximal biclique with anchor on left using only remaining edges."""
    left = {anchor}
    # Right: neighbors of anchor that are incident to remaining edges
    right = set()
    for u, v in remaining:
        if u == anchor:
            right.add(v)
        elif v == anchor:
            right.add(u)
    
    if not right:
        return None, None
    
    # Iteratively refine
    for _ in range(200):
        if time.time() > deadline:
            break
        changed = False
        # Filter right: must be connected to all left via remaining edges
        new_right = set()
        for v in right:
            ok = True
            for u in left:
                if _edge_key(u, v) not in remaining:
                    ok = False
                    break
            if ok:
                new_right.add(v)
        if new_right != right:
            right = new_right
            changed = True
        
        # Filter left: must be connected to all right via remaining edges
        new_left = set()
        for u in left:
            ok = True
            for v in right:
                if _edge_key(u, v) not in remaining:
                    ok = False
                    break
            if ok:
                new_left.add(u)
        if new_left != left:
            left = new_left
            changed = True
        
        if not changed:
            break
    
    if not left or not right:
        return None, None
    
    # Expand left
    for u in range(n):
        if u not in left:
            ok = True
            for v in right:
                if _edge_key(u, v) not in remaining:
                    ok = False
                    break
            if ok:
                left.add(u)
    
    # Expand right
    for v in range(n):
        if v not in right:
            ok = True
            for u in left:
                if _edge_key(u, v) not in remaining:
                    ok = False
                    break
            if ok:
                right.add(v)
    
    return left, right

def _greedy_partition(edge_set, n, deadline, seed):
    """Greedy biclique extraction with best-first edge selection using anchor-based maximal bicliques."""
    rng = random.Random(seed)
    remaining = set(edge_set)
    adj = _build_adjacency(remaining, n)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find vertex with max degree in remaining
        deg = {}
        for u, v in remaining:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        
        if not deg:
            break
        
        # Try multiple anchors, pick best
        sorted_anchors = sorted(deg.keys(), key=lambda x: -deg[x])
        best_left, best_right, best_size = None, None, 0
        
        num_starts = min(20, len(sorted_anchors))
        for anchor in sorted_anchors[:num_starts]:
            if time.time() > deadline:
                break
            left, right = _maximal_biclique_from_anchor(adj, anchor, remaining, n, deadline)
            if left is None:
                continue
            size = len(left) * len(right)
            if size > best_size:
                best_size = size
                best_left = left
                best_right = right
        
        if best_left is None:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard(_edge_key(u, v))
            adj[u].discard(v)
            adj[v].discard(u)
            continue
        
        # Remove covered edges
        for u in best_left:
            for v in best_right:
                key = _edge_key(u, v)
                if key in remaining:
                    remaining.discard(key)
                    adj[u].discard(v)
                    adj[v].discard(u)
        
        bicliques.append([list(best_left), list(best_right)])
    
    # Leftover edges
    for e in remaining:
        bicliques.append([[e[0]], [e[1]]])
    
    return bicliques

def _try_merge(bicliques, edge_set, n, deadline):
    """Try to merge pairs of bicliques if their union forms a valid biclique."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try merging: (L1∪L2, R1∪R2)
                L_new = set(L1) | set(L2)
                R_new = set(R1) | set(R2)
                
                # Check all cross pairs are edges
                valid = True
                for u in L_new:
                    for v in R_new:
                        if u == v or _edge_key(u, v) not in edge_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check coverage matches
                covered_i = set(_edge_key(u, v) for u in L1 for v in R1)
                covered_j = set(_edge_key(u, v) for u in L2 for v in R2)
                covered_new = set(_edge_key(u, v) for u in L_new for v in R_new)
                
                if covered_new == covered_i | covered_j:
                    bicliques[i] = [list(L_new), list(R_new)]
                    bicliques.pop(j)
                    changed = True
                    break
            if changed:
                break
    return bicliques

def _try_split(bicliques, edge_set, n, deadline):
    """Try to split bicliques to potentially allow better merges."""
    # This is a heuristic: if a biclique is large, try splitting it
    # For now, we'll skip this as it's complex and may not help
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    edges = P['edges']
    n = P['n']
    edge_set = set(_edge_key(u, v) for u, v in edges)
    
    best_bicliques = None
    best_count = float('inf')
    
    # Multiple random restarts with different seeds
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        
        bicliques = _greedy_partition(edge_set, n, deadline, seed)
        
        # Try to merge bicliques to reduce count
        bicliques = _try_merge(bicliques, edge_set, n, deadline)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    if best_bicliques is None:
        # Fallback: star partition
        adj = _build_adjacency(edge_set, n)
        bicliques = []
        for v in range(n):
            if adj[v]:
                bicliques.append([[v], list(adj[v])])
        best_bicliques = bicliques
    
    # Ensure types are correct
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result
