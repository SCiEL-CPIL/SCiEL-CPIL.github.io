"""Model-and-optimize biclique partition for the pinned graph."""
import time
import random
import sys
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _validate(bicliques, edges_set):
    if not bicliques:
        return len(edges_set) == 0
    seen = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            if not isinstance(u, int) or isinstance(u, bool):
                return False
        for v in right:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                e = _edge_key(u, v)
                if e not in edges_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edges_set


def _star_partition(adj):
    result = []
    for u in range(len(adj)):
        neighbors = [v for v in adj[u]]
        if neighbors:
            result.append([neighbors, [u]])
    return result


def _maximal_biclique_from_anchor(adj, anchor, remaining, max_iter=200):
    """Build a maximal biclique with anchor on left using only remaining edges."""
    # Precompute remaining edge set as frozenset for O(1) lookup
    rem_set = set(remaining)
    
    left = {anchor}
    # Right: neighbors of anchor that are incident to remaining edges
    right = set()
    for u, v in rem_set:
        if u == anchor:
            right.add(v)
        elif v == anchor:
            right.add(u)
    
    if not right:
        return None, None
    
    # Iteratively refine
    for _ in range(max_iter):
        changed = False
        # Filter right: must be connected to all left via remaining edges
        new_right = set()
        for v in right:
            ok = True
            for u in left:
                if _edge_key(u, v) not in rem_set:
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
                if _edge_key(u, v) not in rem_set:
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
    for u in range(P['n']):
        if u not in left:
            ok = True
            for v in right:
                if _edge_key(u, v) not in rem_set:
                    ok = False
                    break
            if ok:
                left.add(u)
    
    # Expand right
    for v in range(P['n']):
        if v not in right:
            ok = True
            for u in left:
                if _edge_key(u, v) not in rem_set:
                    ok = False
                    break
            if ok:
                right.add(v)
    
    return left, right


def _greedy_biclique_partition(adj, edges_set, deadline):
    remaining = set(edges_set)
    bicliques = []
    
    while remaining and time.time() < deadline:
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
        
        for anchor in sorted_anchors[:20]:
            left, right = _maximal_biclique_from_anchor(adj, anchor, remaining)
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
            continue
        
        # Remove covered edges
        for u in best_left:
            for v in best_right:
                remaining.discard(_edge_key(u, v))
        
        bicliques.append([list(best_left), list(best_right)])
    
    # Leftover edges
    for e in remaining:
        bicliques.append([[e[0]], [e[1]]])
    
    return bicliques


def _try_merge(bicliques, edges_set):
    """Try to merge pairs of bicliques."""
    n = len(bicliques)
    for i in range(n):
        for j in range(i + 1, n):
            li, ri = bicliques[i][0], bicliques[i][1]
            lj, rj = bicliques[j][0], bicliques[j][1]
            
            new_left = set(li) | set(lj)
            new_right = set(ri) | set(rj)
            
            # Check all cross pairs are edges
            valid = True
            for u in new_left:
                for v in new_right:
                    if u == v or _edge_key(u, v) not in edges_set:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check coverage matches
            covered_i = set(_edge_key(u, v) for u in li for v in ri)
            covered_j = set(_edge_key(u, v) for u in lj for v in rj)
            covered_new = set(_edge_key(u, v) for u in new_left for v in new_right)
            
            if covered_new == covered_i | covered_j:
                bicliques[i] = [list(new_left), list(new_right)]
                del bicliques[j]
                return True
    return False


def _local_search(bicliques, edges_set, deadline):
    """Iteratively try to merge or restructure."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try merging
        n = len(bicliques)
        for i in range(n):
            for j in range(i + 1, n):
                li, ri = bicliques[i][0], bicliques[i][1]
                lj, rj = bicliques[j][0], bicliques[j][1]
                
                new_left = set(li) | set(lj)
                new_right = set(ri) | set(rj)
                
                valid = True
                for u in new_left:
                    for v in new_right:
                        if u == v or _edge_key(u, v) not in edges_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                covered_i = set(_edge_key(u, v) for u in li for v in ri)
                covered_j = set(_edge_key(u, v) for u in lj for v in rj)
                covered_new = set(_edge_key(u, v) for u in new_left for v in new_right)
                
                if covered_new == covered_i | covered_j:
                    bicliques[i] = [list(new_left), list(new_right)]
                    del bicliques[j]
                    improved = True
                    break
            if improved:
                break
        
        # Also try: for each biclique, try to move a vertex from one side to the other
        # or try to absorb edges from a neighboring biclique
        if not improved and time.time() < deadline:
            n = len(bicliques)
            for i in range(n):
                li, ri = bicliques[i][0], bicliques[i][1]
                # Try adding a vertex from another biclique's left to our right (or vice versa)
                for j in range(n):
                    if i == j:
                        continue
                    lj, rj = bicliques[j][0], bicliques[j][1]
                    # Try moving a vertex from lj to ri
                    for u in lj:
                        if u in ri:
                            continue
                        # Check if u can be added to ri
                        ok = all(_edge_key(u, v) in edges_set for v in li)
                        if ok:
                            new_ri = ri + [u]
                            new_lj = [x for x in lj if x != u]
                            if new_lj:
                                # Check new biclique j is still valid
                                ok2 = all(_edge_key(a, b) in edges_set for a in new_lj for b in rj)
                                if ok2:
                                    # Check no overlap in covered edges
                                    cov_i = set(_edge_key(a, b) for a in li for b in new_ri)
                                    cov_j = set(_edge_key(a, b) for a in new_lj for b in rj)
                                    if cov_i & cov_j:
                                        continue
                                    bicliques[i] = [li, new_ri]
                                    bicliques[j] = [new_lj, rj]
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        
        # Try to split a biclique into two if it helps overall count
        # (This is rare but can help in some cases)
        if not improved and time.time() < deadline:
            n = len(bicliques)
            for i in range(n):
                li, ri = bicliques[i][0], bicliques[i][1]
                if len(li) < 2 or len(ri) < 2:
                    continue
                # Try removing one vertex from left and see if remaining is still valid
                for u in li:
                    new_li = [x for x in li if x != u]
                    if not new_li:
                        continue
                    # Check if [new_li, ri] is still a valid biclique
                    ok = all(_edge_key(a, b) in edges_set for a in new_li for b in ri)
                    if not ok:
                        continue
                    # The removed vertex u forms a star with its neighbors in ri
                    # But we need to check if this actually reduces total bicliques
                    # Only do this if it allows merging with other bicliques
                    # For now, skip this complex operation
                    pass
    return bicliques


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    adj = [set() for _ in range(P['n'])]
    edges_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edges_set.add(_edge_key(u, v))
    
    # Fallback
    fallback = _star_partition(adj)
    best = fallback
    best_count = len(fallback)
    
    # Multiple restarts with different anchor orderings
    random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        # Shuffle anchor priority for diversity
        anchors = list(range(P['n']))
        if attempt > 0:
            random.shuffle(anchors)
        
        # Greedy with modified anchor selection
        remaining = set(edges_set)
        bicliques = []
        
        while remaining and time.time() < deadline - 5:
            deg = {}
            for u, v in remaining:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            if not deg:
                break
            
            # Select anchor: highest degree, or random from top-k
            if attempt == 0:
                sorted_anchors = sorted(deg.keys(), key=lambda x: -deg[x])
                anchor_candidates = sorted_anchors[:15]
            else:
                sorted_anchors = sorted(deg.keys(), key=lambda x: -deg[x])
                top_k = min(15, len(sorted_anchors))
                anchor_candidates = sorted_anchors[:top_k]
                random.shuffle(anchor_candidates)
            
            best_left, best_right, best_size = None, None, 0
            
            for anchor in anchor_candidates:
                left, right = _maximal_biclique_from_anchor(adj, anchor, remaining)
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
                continue
            
            for u in best_left:
                for v in best_right:
                    remaining.discard(_edge_key(u, v))
            
            bicliques.append([list(best_left), list(best_right)])
        
        for e in remaining:
            bicliques.append([[e[0]], [e[1]]])
        
        # Local search
        if time.time() < deadline - 10:
            bicliques = _local_search(bicliques, edges_set, deadline - 10)
        
        if _validate(bicliques, edges_set) and len(bicliques) < best_count:
            best = bicliques
            best_count = len(bicliques)
    
    # Final validation
    if not _validate(best, edges_set):
        best = fallback
    
    # Ensure correct types
    result = []
    for left, right in best:
        result.append([[int(u) for u in left], [int(v) for v in right]])
    
    return result
