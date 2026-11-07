"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_rem_adj(remaining_edges, n):
    rem_adj = [set() for _ in range(n)]
    for e in remaining_edges:
        u, v = tuple(e)
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    return rem_adj

def _expand_biclique(start_v, remaining_edges, rem_adj, n, deadline):
    """Expand a biclique starting from vertex start_v on the remaining graph.
    Returns (L, R) as sets, or (None, None) if invalid/timeout."""
    if time.time() > deadline:
        return None, None
    
    # Initial R: all neighbors of start_v in remaining graph
    R = set(rem_adj[start_v])
    if not R:
        return None, None
    
    L = {start_v}
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # Expand L: find w not in L∪R such that R ⊆ rem_adj[w]
        best_w_L = None
        best_score_L = -1
        for w in range(n):
            if w in L or w in R:
                continue
            if R.issubset(rem_adj[w]):
                score = len(rem_adj[w])
                if score > best_score_L:
                    best_score_L = score
                    best_w_L = w
        if best_w_L is not None:
            L.add(best_w_L)
            changed = True
        
        # Expand R: find w not in L∪R such that L ⊆ rem_adj[w]
        best_w_R = None
        best_score_R = -1
        for w in range(n):
            if w in L or w in R:
                continue
            if L.issubset(rem_adj[w]):
                score = len(rem_adj[w])
                if score > best_score_R:
                    best_score_R = score
                    best_w_R = w
        if best_w_R is not None:
            R.add(best_w_R)
            changed = True
    
    # Verify all cross pairs are in remaining_edges
    for l in L:
        for r in R:
            if r not in rem_adj[l]:
                return None, None
    
    return L, R

def _greedy_partition(remaining_edges, n, deadline, num_candidates=5, randomize=False):
    """Greedy biclique partition using adjacency sets. Returns list of bicliques or None."""
    current_remaining = set(remaining_edges)
    bicliques = []
    
    while current_remaining:
        if time.time() > deadline:
            return None
        
        # Compute degrees in remaining graph
        deg = [0] * n
        for e in current_remaining:
            u, v = tuple(e)
            deg[u] += 1
            deg[v] += 1
        
        # Build adjacency for remaining graph
        rem_adj = _build_rem_adj(current_remaining, n)
        
        # Pick candidate starting vertices
        if randomize:
            # Random sampling from high-degree vertices
            high_deg = [v for v in range(n) if deg[v] > 0]
            random.shuffle(high_deg)
            candidates = high_deg[:min(num_candidates, len(high_deg))]
        else:
            sorted_verts = sorted(range(n), key=lambda x: deg[x], reverse=True)
            candidates = [v for v in sorted_verts[:num_candidates] if deg[v] > 0]
        
        if not candidates:
            break
        
        # Try each candidate, pick the one giving the largest biclique
        best_L = None
        best_R = None
        best_removed = 0
        
        for v_start in candidates:
            L, R = _expand_biclique(v_start, current_remaining, rem_adj, n, deadline)
            if L is None:
                continue
            # Count edges covered
            removed = 0
            for l in L:
                for r in R:
                    if frozenset((l, r)) in current_remaining:
                        removed += 1
            if removed > best_removed:
                best_removed = removed
                best_L = L
                best_R = R
        
        if best_removed == 0 or best_L is None:
            # Fallback: single edge
            e = next(iter(current_remaining))
            u, v = tuple(e)
            bicliques.append([[u], [v]])
            current_remaining.remove(e)
            continue
        
        # Remove cross edges from current_remaining
        for l in best_L:
            for r in best_R:
                e = frozenset((l, r))
                if e in current_remaining:
                    current_remaining.remove(e)
        
        bicliques.append([sorted(best_L), sorted(best_R)])
    
    return bicliques

def _try_merge(bicliques, host_adj, n, deadline):
    """Try to merge compatible bicliques to reduce count."""
    if len(bicliques) <= 1:
        return bicliques
    
    merged = True
    current = bicliques[:]
    
    while merged and time.time() < deadline:
        merged = False
        new_bicliques = []
        i = 0
        while i < len(current):
            L1 = set(current[i][0])
            R1 = set(current[i][1])
            j = i + 1
            found_merge = False
            while j < len(current):
                L2 = set(current[j][0])
                R2 = set(current[j][1])
                
                # Try all four merge orientations:
                # (L1∪L2, R1∪R2), (L1∪R2, R1∪L2), (R1∪L2, L1∪R2), (R1∪R2, L1∪L2)
                orientations = [
                    (L1 | L2, R1 | R2),
                    (L1 | R2, R1 | L2),
                    (R1 | L2, L1 | R2),
                    (R1 | R2, L1 | L2),
                ]
                
                for new_L, new_R in orientations:
                    if not new_L or not new_R:
                        continue
                    if new_L & new_R:
                        continue
                    # Check all cross pairs are host edges
                    valid = True
                    for l in new_L:
                        for r in new_R:
                            if r not in host_adj[l]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        new_bicliques.append([sorted(new_L), sorted(new_R)])
                        found_merge = True
                        break
                if found_merge:
                    break
                j += 1
            if not found_merge:
                new_bicliques.append([sorted(L1), sorted(R1)])
            i += 1
        
        if len(new_bicliques) < len(current):
            current = new_bicliques
            merged = True
        else:
            merged = False
    
    return current

def _local_search(bicliques, host_adj, n, remaining_edges, deadline):
    """Try to improve the partition by re-extracting or swapping."""
    if time.time() > deadline:
        return bicliques
    
    current = bicliques[:]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing one vertex from a side and re-extracting
        for bi in range(len(current)):
            if time.time() > deadline:
                break
            L = list(current[bi][0])
            R = list(current[bi][1])
            
            # Try moving a vertex from L to R or vice versa, or removing it
            for side, vertices in [('L', L), ('R', R)]:
                for v in vertices:
                    if time.time() > deadline:
                        break
                    # Remove v from its side
                    if side == 'L':
                        new_L = [x for x in L if x != v]
                        new_R = R
                    else:
                        new_L = L
                        new_R = [x for x in R if x != v]
                    
                    if not new_L or not new_R:
                        continue
                    
                    # Check validity
                    valid = True
                    for l in new_L:
                        for r in new_R:
                            if r not in host_adj[l]:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # This is a valid smaller biclique; we could potentially
                        # re-extract the freed edges into other bicliques
                        # For now, just accept the smaller biclique
                        # Actually, this doesn't help reduce count directly.
                        # Skip this direction for now.
                        pass
        
        # Try merging again
        merged = _try_merge(current, host_adj, n, deadline)
        if len(merged) < len(current):
            current = merged
            improved = True
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build static adjacency sets for host graph
    host_adj = _build_adj(n, edges)
    
    # Initialize remaining edges
    remaining_edges = set()
    for u, v in edges:
        if u < v:
            remaining_edges.add(frozenset((u, v)))
        else:
            remaining_edges.add(frozenset((v, u)))
    
    best_solution = None
    best_count = float('inf')
    
    # Run multiple strategies
    strategies = [
        (5, False),   # 5 candidates, deterministic
        (8, False),   # 8 candidates, deterministic
        (5, True),    # 5 candidates, randomized
        (8, True),    # 8 candidates, randomized
        (3, False),   # 3 candidates, deterministic (faster, more restarts)
    ]
    
    for num_cand, rand in strategies:
        if time.time() > deadline:
            break
        
        # Run multiple attempts per strategy
        max_attempts = 10 if num_cand <= 5 else 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            bicliques = _greedy_partition(remaining_edges, n, deadline, num_cand, rand)
            if bicliques is None:
                continue
            
            # Verify solution
            covered_edges = set()
            valid = True
            for left, right in bicliques:
                for l in left:
                    for r in right:
                        e = frozenset((l, r))
                        if e in covered_edges:
                            valid = False
                            break
                        covered_edges.add(e)
                        if e not in remaining_edges:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            if covered_edges != remaining_edges:
                continue
            
            # Try to merge
            bicliques = _try_merge(bicliques, host_adj, n, deadline)
            
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_solution = bicliques
                
            if best_count == 1:
                break
        
        if best_count == 1:
            break
    
    if best_solution is None:
        # Fallback: star decomposition
        bicliques = []
        for u in range(n):
            R_u = [v for v in host_adj[u] if v > u]
            if R_u:
                bicliques.append([[u], sorted(R_u)])
        best_solution = bicliques
        best_count = len(bicliques)
    
    # Convert to required format
    final_result = []
    for bic in best_solution:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        final_result.append([left, right])
    
    return final_result

# EVOLVE-BLOCK-END
