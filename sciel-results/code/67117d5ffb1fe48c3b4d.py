"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_biclique(adj, remaining_edges, anchor):
    """Find a large biclique anchored at `anchor` using remaining edges."""
    n = P['n']
    # right = neighbors of anchor that still have edges
    right = set()
    for v in adj[anchor]:
        if (min(anchor, v), max(anchor, v)) in remaining_edges:
            right.add(v)
    if not right:
        return None, None
    
    # left starts as {anchor}
    left = {anchor}
    
    # Iteratively grow left: add vertices that connect to all of right
    changed = True
    while changed:
        changed = False
        for v in list(adj[anchor]):
            if v in left:
                continue
            # Check if v connects to all of right
            can_add = True
            for w in right:
                if w not in adj[v]:
                    can_add = False
                    break
            if can_add:
                # Also need v to have remaining edges to right
                has_remaining = False
                for w in right:
                    if (min(v, w), max(v, w)) in remaining_edges:
                        has_remaining = True
                        break
                if has_remaining:
                    left.add(v)
                    changed = True
    
    # Now shrink right to only those connected to all of left
    new_right = set()
    for w in right:
        ok = True
        for u in left:
            if w not in adj[u]:
                ok = False
                break
        if ok:
            new_right.add(w)
    right = new_right
    
    if not left or not right:
        return None, None
    
    return list(left), list(right)

def _greedy_partition(adj, order, remaining_edges=None):
    """Greedy partition following vertex order."""
    n = P['n']
    if remaining_edges is None:
        remaining = set()
        for u, v in P['edges']:
            remaining.add((min(u, v), max(u, v)))
    else:
        remaining = set(remaining_edges)
    
    bicliques = []
    
    for anchor in order:
        if not remaining:
            break
        # Check if anchor still has remaining edges
        has_edge = False
        for v in adj[anchor]:
            if (min(anchor, v), max(anchor, v)) in remaining:
                has_edge = True
                break
        if not has_edge:
            continue
        
        left, right = _find_max_biclique(adj, remaining, anchor)
        if left is None:
            # Fallback: single vertex biclique
            right = [v for v in adj[anchor] if (min(anchor, v), max(anchor, v)) in remaining]
            if right:
                left = [anchor]
                right = right
            else:
                continue
        else:
            left = list(left)
            right = list(right)
        
        # Remove covered edges
        for u in left:
            for v in right:
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
        
        bicliques.append([left, right])
    
    # Handle any remaining edges with singleton bicliques
    if remaining:
        for e in list(remaining):
            u, v = e
            bicliques.append([[u], [v]])
            remaining.discard(e)
    
    return bicliques

def _validate(bicliques):
    """Check if biclique partition is valid."""
    n = P['n']
    adj = _build_adj()
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    for u, v in P['edges']:
        e = (min(u, v), max(u, v))
        if e not in covered:
            return False
    return True

def _merge_pass(bicliques, adj):
    """Try to merge pairs of bicliques."""
    n = P['n']
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(bicliques):
            j = i + 1
            while j < len(bicliques):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try merge: [L1∪L2, R1∪R2]
                new_L = set(L1) | set(L2)
                new_R = set(R1) | set(R2)
                
                # Check if [new_L, new_R] is a valid biclique
                valid = True
                for u in new_L:
                    for v in new_R:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Check that all edges in new_L x new_R are covered by these two bicliques
                    # i.e., no edge in new_L x new_R was covered by other bicliques
                    # This is complex; skip for safety
                    # Just do simple merge if valid
                    bicliques[i] = [list(new_L), list(new_R)]
                    del bicliques[j]
                    changed = True
                    break
                j += 1
            if changed:
                break
            i += 1
    return bicliques

def _local_search(bicliques, adj, max_iterations=100):
    """Local search: try to reduce biclique count by moving vertices between bicliques."""
    n = P['n']
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(len(bicliques)):
            for j in range(i + 1, len(bicliques)):
                if i >= len(bicliques) or j >= len(bicliques):
                    break
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try moving a vertex from L1 to L2 or R1 to R2
                # Check if we can merge by moving vertices
                # Try: move all of L1 into L2, and all of R1 into R2
                new_L2 = set(L2) | set(L1)
                new_R2 = set(R2) | set(R1)
                
                # Check if [new_L2, new_R2] is valid
                valid = True
                for u in new_L2:
                    for v in new_R2:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Check that all edges in new_L2 x new_R2 are only covered by these two bicliques
                    # For now, just do the merge if valid
                    bicliques[i] = [list(set(L1) | set(L2)), list(set(R1) | set(R2))]
                    del bicliques[j]
                    improved = True
                    break
            
            if improved:
                break
        
        # Try splitting and recombining
        for i in range(len(bicliques)):
            L1, R1 = bicliques[i]
            if len(L1) > 1:
                # Try moving one vertex from L1 to another biclique
                for v in L1:
                    for j in range(len(bicliques)):
                        if i >= len(bicliques) or j >= len(bicliques):
                            break
                        if i == j:
                            continue
                        L2, R2 = bicliques[j]
                        # Check if v can be added to L2
                        can_add = True
                        for w in R2:
                            if w not in adj[v]:
                                can_add = False
                                break
                        if can_add and v not in L2:
                            # Check if removing v from L1 and adding to L2 keeps both valid
                            # For simplicity, skip this for now
                            pass
    
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # Compute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Strategy 1: Greedy with vertices in degree order (descending)
    order1 = sorted(range(n), key=lambda x: -degrees[x])
    result1 = _greedy_partition(adj, order1)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 10:
        # Not enough time, return best so far
        if _validate(result1):
            return [[list(l), list(r)] for l, r in result1]
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(n) if any(a == u for a, v in P['edges'])]
    
    # Strategy 2: Greedy with vertices in reverse degree order
    order2 = sorted(range(n), key=lambda x: degrees[x])
    result2 = _greedy_partition(adj, order2)
    
    # Strategy 3: Multiple random restarts
    best_result = min(result1, result2, key=len)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 20:
        # Try merge on best
        best_result = _merge_pass(best_result, adj)
        if _validate(best_result):
            return [[list(l), list(r)] for l, r in best_result]
        return [[list(l), list(r)] for l, r in result1]
    
    # Try more random orderings
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        random.seed(attempt)
        order = list(range(n))
        random.shuffle(order)
        result = _greedy_partition(adj, order)
        if len(result) < len(best_result):
            best_result = result
    
    # Try merge pass on best
    if time.time() < deadline - 2:
        merged = _merge_pass(best_result, adj)
        if _validate(merged) and len(merged) <= len(best_result):
            best_result = merged
    
    # Try local search
    if time.time() < deadline - 2:
        local_result = _local_search(best_result, adj)
        if _validate(local_result) and len(local_result) <= len(best_result):
            best_result = local_result
    
    # Final validation
    if not _validate(best_result):
        # Fallback to seed
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(n) if any(a == u for a, v in P['edges'])]
    
    return [[list(l), list(r)] for l, r in best_result]

# EVOLVE-BLOCK-END
