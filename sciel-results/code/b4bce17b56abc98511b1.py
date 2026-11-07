"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_extract(remaining_edges, adj, n, anchor=None):
    """Extract a single biclique from remaining edges using greedy growth."""
    if not remaining_edges:
        return None, set()
    
    # Build adjacency for remaining edges only
    rem_adj = [set() for _ in range(n)]
    rem_deg = [0] * n
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        rem_deg[u] += 1
        rem_deg[v] += 1
    
    # Choose anchor
    if anchor is None:
        # Pick vertex with highest remaining degree
        candidates = [(rem_deg[i], i) for i in range(n) if rem_deg[i] > 0]
        if not candidates:
            return None, set()
        candidates.sort(reverse=True)
        anchor = candidates[0][1]
    
    if rem_deg[anchor] == 0:
        return None, set()
    
    # Greedy biclique growth starting from anchor
    left = {anchor}
    right = set(rem_adj[anchor])
    
    # Iteratively grow
    changed = True
    while changed:
        changed = False
        # Try to add vertices to right
        new_right = set()
        for v in list(right):
            for w in rem_adj[v]:
                if w not in right and w not in left:
                    # Check if w connects to all in left
                    if all(w in rem_adj[u] for u in left):
                        new_right.add(w)
        if new_right:
            right |= new_right
            changed = True
        
        # Try to add vertices to left
        new_left = set()
        for v in list(left):
            for w in rem_adj[v]:
                if w not in left and w not in right:
                    # Check if w connects to all in right
                    if all(w in rem_adj[u] for u in right):
                        new_left.add(w)
        if new_left:
            left |= new_left
            changed = True
    
    # Collect edges in this biclique
    biclique_edges = set()
    for u in left:
        for v in right:
            if v in rem_adj[u]:
                biclique_edges.add(frozenset((u, v)))
    
    return (list(left), list(right)), biclique_edges

def _try_merge(bicliques):
    """Try to merge compatible bicliques. Returns merged list."""
    if len(bicliques) <= 1:
        return bicliques
    
    merged = True
    result = list(bicliques)
    while merged:
        merged = False
        new_result = []
        i = 0
        while i < len(result):
            j = i + 1
            while j < len(result):
                L1, R1 = result[i]
                L2, R2 = result[j]
                
                # Check if union is a valid biclique
                # All pairs in L1xR2, L2xR1, L1xR1, L2xR2 must be edges
                # But we need to check against the original graph
                # Actually, we need to verify that every cross-pair in the union is an edge
                # This is expensive, so let's do a quick check first
                # The union biclique would be [L1|L2, R1|R2]
                # But we need L1∩R2, L2∩R1 to be empty for it to be a proper biclique
                # Actually no - a biclique just needs left and right to be disjoint and all cross pairs to be edges
                
                # Check disjointness
                if not (set(L1) & set(R2)) and not (set(L2) & set(R1)):
                    # Check all cross pairs are edges
                    # This is O(|L1||R2| + |L2||R1|)
                    ok = True
                    for u in L1:
                        for v in R2:
                            if v not in adj_cache[u]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        for u in L2:
                            for v in R1:
                                if v not in adj_cache[u]:
                                    ok = False
                                    break
                        if not ok:
                            break
                    
                    if ok:
                        # Merge
                        new_L = list(set(L1) | set(L2))
                        new_R = list(set(R1) | set(R2))
                        # Ensure disjoint
                        if not (set(new_L) & set(new_R)):
                            result[i] = (new_L, new_R)
                            result.pop(j)
                            merged = True
                            continue
                j += 1
            new_result.append(result[i])
            i += 1
        result = new_result
    
    return result

# Global adjacency cache
adj_cache = None

def construct():
    global adj_cache
    start_time = time.time()
    n = P['n']
    edges = P['edges']
    
    # Build full adjacency
    adj_cache = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj_cache[u].add(v)
        adj_cache[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    best_bicliques = None
    best_count = float('inf')
    
    # Try multiple random restarts
    rng = random.Random(42)
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Shuffle edges for different ordering
        shuffled_edges = list(edges)
        if restart > 0:
            rng.shuffle(shuffled_edges)
        
        remaining = set(frozenset(e) for e in shuffled_edges)
        bicliques = []
        
        while remaining:
            if time.time() - start_time > DEADLINE:
                break
            
            # Try multiple anchors
            best_bq = None
            best_bq_edges = None
            best_bq_size = 0
            
            # Top candidates by degree in remaining
            rem_deg = [0] * n
            for e in remaining:
                u, v = tuple(e)
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            candidates = [(rem_deg[i], i) for i in range(n) if rem_deg[i] > 0]
            candidates.sort(reverse=True)
            top_anchors = [c[1] for c in candidates[:20]]
            
            for anchor in top_anchors:
                bq, bq_edges = _greedy_extract(list(remaining), adj_cache, n, anchor=anchor)
                if bq and bq_edges and len(bq_edges) > best_bq_size:
                    best_bq = bq
                    best_bq_edges = bq_edges
                    best_bq_size = len(bq_edges)
            
            if best_bq is None:
                # Fallback: just take one edge
                e = next(iter(remaining))
                u, v = tuple(e)
                best_bq = ([u], [v])
                best_bq_edges = {e}
            
            bicliques.append(best_bq)
            remaining -= best_bq_edges
        
        if not remaining and bicliques:
            # Merge pass
            merged = _try_merge(bicliques)
            if len(merged) < best_count:
                best_count = len(merged)
                best_bicliques = merged
        
        if time.time() - start_time > DEADLINE:
            break
    
    # If no valid result found, use the seed program's result
    if best_bicliques is None:
        best_bicliques = [[[u], [v for a, v in P['edges'] if a == u]] for u in range(n) if any(a == u for a, v in P['edges'])]
    
    # Convert to proper types
    result = []
    for bq in best_bicliques:
        left = [int(x) for x in bq[0]]
        right = [int(x) for x in bq[1]]
        result.append([left, right])
    
    return result

# EVOLVE-BLOCK-END
