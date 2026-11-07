import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = 75.0  # Safety margin (90s limit, leave 15s buffer)
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # remaining_edges as set of (u, v) with u < v
    remaining_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        remaining_edges.add((u, v))
    
    def can_extend(L, R, candidate, side):
        """Check if candidate can be added to side while maintaining biclique property."""
        if side == 'L':
            # candidate must connect to all in R
            return all(candidate in adj[r] for r in R)
        else:
            # candidate must connect to all in L
            return all(candidate in adj[l] for l in L)
    
    def expand_biclique(L, R, remaining):
        """Expand L and R to be maximal w.r.t. remaining edges using greedy heuristic."""
        changed = True
        while changed:
            changed = False
            
            # Try adding to L - pick vertex that maximizes potential R additions
            best_l = None
            best_l_score = -1
            for u_prime in range(n):
                if u_prime in L:
                    continue
                if not all(u_prime in adj[r] for r in R):
                    continue
                # Score: number of new R candidates enabled
                score = 0
                for v_prime in range(n):
                    if v_prime in R:
                        continue
                    if all(v_prime in adj[l] for l in L) and v_prime in adj[u_prime]:
                        if (min(u_prime, v_prime), max(u_prime, v_prime)) in remaining:
                            score += 1
                if score > best_l_score:
                    best_l_score = score
                    best_l = u_prime
            if best_l is not None and best_l_score > 0:
                L.add(best_l)
                changed = True
            
            # Try adding to R - pick vertex that maximizes potential L additions
            best_r = None
            best_r_score = -1
            for v_prime in range(n):
                if v_prime in R:
                    continue
                if not all(v_prime in adj[l] for l in L):
                    continue
                # Score: number of new L candidates enabled
                score = 0
                for u_prime in range(n):
                    if u_prime in L:
                        continue
                    if all(u_prime in adj[r] for r in R) and u_prime in adj[v_prime]:
                        if (min(u_prime, v_prime), max(u_prime, v_prime)) in remaining:
                            score += 1
                if score > best_r_score:
                    best_r_score = score
                    best_r = v_prime
            if best_r is not None and best_r_score > 0:
                R.add(best_r)
                changed = True
        return L, R
    
    def count_covered(L, R, remaining):
        """Count how many edges in L x R are in remaining."""
        count = 0
        for l in L:
            for r in R:
                if l != r and (min(l, r), max(l, r)) in remaining:
                    count += 1
        return count
    
    def find_best_biclique(remaining):
        """Find the biclique that covers the most edges from remaining using multi-anchor strategy."""
        best_L = None
        best_R = None
        best_edges = 0
        
        # Get top anchors by remaining degree
        deg = [0] * n
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        
        # Sort vertices by remaining degree descending
        sorted_vertices = sorted(range(n), key=lambda x: -deg[x])
        
        # Try top anchors as seeds
        tried = set()
        for idx, anchor in enumerate(sorted_vertices[:20]):
            if time.time() > deadline:
                break
            # Try each neighbor of anchor as the other seed
            for neighbor in sorted(adj[anchor], key=lambda x: -deg[x])[:10]:
                if neighbor in tried:
                    continue
                key = (min(anchor, neighbor), max(anchor, neighbor))
                if key not in remaining:
                    continue
                tried.add(key)
                
                L = {anchor}
                R = {neighbor}
                L, R = expand_biclique(L, R, remaining)
                ec = count_covered(L, R, remaining)
                if ec > best_edges:
                    best_edges = ec
                    best_L = set(L)
                    best_R = set(R)
        
        return best_L, best_R, best_edges
    
    bicliques = []
    
    # Phase 1: Greedy partition with best-biclique selection
    while remaining_edges and time.time() < deadline:
        L, R, ec = find_best_biclique(remaining_edges)
        
        if L is None or R is None or ec == 0:
            # Fallback: pick any edge
            u, v = next(iter(remaining_edges))
            L = {u}
            R = {v}
        
        # Remove covered edges
        for l in L:
            for r in R:
                if l != r:
                    key = (min(l, r), max(l, r))
                    if key in remaining_edges:
                        remaining_edges.remove(key)
        
        bicliques.append([list(L), list(R)])
    
    # Phase 2: Fallback for any remaining edges
    if remaining_edges:
        for u, v in remaining_edges:
            bicliques.append([[u], [v]])
    
    # Phase 3: Aggressive merge pass - try to merge compatible bicliques
    # Two bicliques can be merged if their union forms a valid biclique
    merged = True
    while merged and time.time() < deadline - 5:
        merged = False
        for i in range(len(bicliques)):
            if time.time() > deadline - 5:
                break
            L1, R1 = bicliques[i][0], bicliques[i][1]
            L1_set, R1_set = set(L1), set(R1)
            for j in range(i+1, len(bicliques)):
                L2, R2 = bicliques[j][0], bicliques[j][1]
                L2_set, R2_set = set(L2), set(R2)
                
                # Check if we can merge: L1∪L2 x R1∪R2 must all be edges
                new_L = L1_set | L2_set
                new_R = R1_set | R2_set
                
                valid = True
                for l in new_L:
                    for r in new_R:
                        if l == r:
                            continue
                        if r not in adj[l]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Merge: replace bicliques[i] with merged, remove j
                bicliques[i] = [list(new_L), list(new_R)]
                bicliques.pop(j)
                merged = True
                break
            if merged:
                break
    
    # Phase 4: Final optimization - try to absorb single-edge bicliques into others
    # A single-edge biclique [[u],[v]] can be absorbed if u can join L and v can join R
    # of an existing biclique, or vice versa
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline - 2:
                break
            L_i, R_i = bicliques[i][0], bicliques[i][1]
            if len(L_i) == 1 and len(R_i) == 1:
                u, v = L_i[0], R_i[0]
                for j in range(len(bicliques)):
                    if i == j:
                        continue
                    L_j, R_j = bicliques[j][0], bicliques[j][1]
                    L_j_set, R_j_set = set(L_j), set(R_j)
                    
                    # Try adding u to L_j and v to R_j
                    if u not in L_j_set and v not in R_j_set:
                        # Check if u connects to all in R_j
                        if all(u in adj[r] for r in R_j):
                            # Check if v connects to all in L_j
                            if all(v in adj[l] for l in L_j):
                                # Check if u connects to all in L_j (within-side allowed, so no constraint)
                                # Check if v connects to all in R_j (within-side allowed, so no constraint)
                                # But we need u-v to be an edge (cross-edge)
                                if v in adj[u]:
                                    # Absorb: add u to L_j, v to R_j
                                    L_j.append(u)
                                    R_j.append(v)
                                    bicliques.pop(i)
                                    changed = True
                                    break
                    
                    # Try adding v to L_j and u to R_j
                    if v not in L_j_set and u not in R_j_set:
                        if all(v in adj[r] for r in R_j):
                            if all(u in adj[l] for l in L_j):
                                if u in adj[v]:
                                    L_j.append(v)
                                    R_j.append(u)
                                    bicliques.pop(i)
                                    changed = True
                                    break
            if changed:
                break
    
    # Ensure types are correct: vertices are Python int, sides are lists
    result = []
    for bic in bicliques:
        L = [int(v) for v in bic[0]]
        R = [int(v) for v in bic[1]]
        result.append([L, R])
    
    return result
# EVOLVE-BLOCK-END
