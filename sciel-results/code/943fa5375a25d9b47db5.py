import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Helper: expand a biclique (L_mask, R_mask) given remaining masks
    # Returns (L_set, R_set, covered_count)
    def expand_biclique(start_u, start_v, rem_mask):
        L = {start_u}
        R = {start_v}
        changed = True
        while changed:
            changed = False
            # Compute mask of R
            mask_R = 0
            for rv in R:
                mask_R |= (1 << rv)
            
            # Expand L: find u' not in L such that rem_mask[u'] covers all of R
            for u_prime in range(n):
                if u_prime in L:
                    continue
                if (rem_mask[u_prime] & mask_R) == mask_R:
                    L.add(u_prime)
                    changed = True
            
            # Compute mask of L
            mask_L = 0
            for lu in L:
                mask_L |= (1 << lu)
            
            # Expand R: find v' not in R such that all L connect to v' in rem_mask
            for v_prime in range(n):
                if v_prime in R:
                    continue
                valid = True
                for lu in L:
                    if not (rem_mask[lu] & (1 << v_prime)):
                        valid = False
                        break
                if valid:
                    R.add(v_prime)
                    changed = True
        
        return L, R, len(L) * len(R)
    
    # Greedy with multiple restarts and degree-ordered candidate generation
    def try_greedy(restarts=200):
        nonlocal best_bicliques, best_count
        
        # Sort vertices by degree for better candidate ordering
        degrees = [bin(adj[u]).count('1') for u in range(n)]
        vert_order = sorted(range(n), key=lambda u: -degrees[u])
        
        for restart in range(restarts):
            if time.time() > deadline:
                return
            
            # Initialize remaining masks
            rem_mask = adj[:]
            edges_left = len(edges)
            current_bicliques = []
            
            while edges_left > 0:
                if time.time() > deadline:
                    return
                
                best_local = None
                best_size = 0
                
                # Generate candidates: pick high-degree vertices first with some randomness
                candidates = []
                for u in vert_order:
                    if rem_mask[u] == 0:
                        continue
                    mask = rem_mask[u]
                    # Get set bits
                    vs = []
                    m = mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        vs.append(v)
                        m ^= lsb
                    # Take top few by degree
                    vs.sort(key=lambda v: -degrees[v])
                    for v in vs[:5]:
                        candidates.append((u, v))
                
                if not candidates:
                    break
                
                # Shuffle for diversity but keep bias toward high-degree
                random.shuffle(candidates)
                
                for u, v in candidates[:15]:
                    if time.time() > deadline:
                        return
                    if not (rem_mask[u] & (1 << v)):
                        continue
                    
                    L, R, size = expand_biclique(u, v, rem_mask)
                    if size > best_size:
                        best_size = size
                        best_local = (L, R)
                
                if not best_local:
                    # Fallback: find any remaining edge
                    found = False
                    for u in range(n):
                        if rem_mask[u]:
                            m = rem_mask[u]
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            if (rem_mask[v] & (1 << u)):
                                best_local = ({u}, {v})
                                best_size = 1
                                found = True
                                break
                    if not found:
                        break
                        
                L, R = best_local
                current_bicliques.append([list(L), list(R)])
                
                # Update rem_mask: remove all edges between L and R
                mask_R = 0
                for v in R:
                    mask_R |= (1 << v)
                mask_L = 0
                for u in L:
                    mask_L |= (1 << u)
                
                for u in L:
                    rem_mask[u] &= ~mask_R
                for v in R:
                    rem_mask[v] &= ~mask_L
                    
                edges_left -= best_size

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                if best_count == 1:
                    return

    # First, create a proper seed that covers all edges
    # Use a simple greedy to get a valid baseline
    rem_mask = adj[:]
    seed_bicliques = []
    edges_left = len(edges)
    while edges_left > 0:
        # Find a vertex with remaining edges
        u = None
        for v in range(n):
            if rem_mask[v]:
                u = v
                break
        if u is None:
            break
        m = rem_mask[u]
        lsb = m & -m
        v = lsb.bit_length() - 1
        L, R, size = expand_biclique(u, v, rem_mask)
        seed_bicliques.append([list(L), list(R)])
        mask_R = 0
        for rv in R:
            mask_R |= (1 << rv)
        mask_L = 0
        for lu in L:
            mask_L |= (1 << lu)
        for lu in L:
            rem_mask[lu] &= ~mask_R
        for rv in R:
            rem_mask[rv] &= ~mask_L
        edges_left -= size
    
    best_bicliques = seed_bicliques
    best_count = len(seed_bicliques)
    
    # Run greedy optimization
    try_greedy()
    
    # Local improvement: try to merge or replace bicliques
    # For each pair of bicliques, check if they can be merged
    def try_merge_improvement():
        nonlocal best_bicliques, best_count
        if len(best_bicliques) < 2:
            return
        
        improved = True
        while improved:
            improved = False
            for i in range(len(best_bicliques)):
                if time.time() > deadline:
                    return
                for j in range(i + 1, len(best_bicliques)):
                    # Try to merge biclique i and j
                    L_i = set(best_bicliques[i][0])
                    R_i = set(best_bicliques[i][1])
                    L_j = set(best_bicliques[j][0])
                    R_j = set(best_bicliques[j][1])
                    
                    # Check if we can form a larger biclique
                    # Option 1: L = L_i ∪ L_j, R = R_i ∩ R_j
                    # Option 2: L = L_i ∩ L_j, R = R_i ∪ R_j
                    # Option 3: L = L_i ∪ L_j, R = R_i ∪ R_j (check validity)
                    
                    # Try option 3 first (most aggressive)
                    L_new = L_i | L_j
                    R_new = R_i | R_j
                    valid = True
                    for u in L_new:
                        for v in R_new:
                            if not (adj[u] & (1 << v)):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(L_new) > 0 and len(R_new) > 0:
                        # Check that this covers all edges from both bicliques
                        # and doesn't add edges that aren't in the graph
                        # The validity check above ensures all cross edges exist
                        new_bicliques = []
                        for k in range(len(best_bicliques)):
                            if k != i and k != j:
                                new_bicliques.append(best_bicliques[k])
                        new_bicliques.append([list(L_new), list(R_new)])
                        
                        # Verify this is a valid partition
                        # Check all edges are covered exactly once
                        covered_edges = set()
                        for b in new_bicliques:
                            L_b = set(b[0])
                            R_b = set(b[1])
                            for u in L_b:
                                for v in R_b:
                                    covered_edges.add((min(u,v), max(u,v)))
                        
                        if covered_edges == {(min(u,v), max(u,v)) for u,v in edges}:
                            best_bicliques = new_bicliques
                            best_count = len(new_bicliques)
                            improved = True
                            break
                if improved:
                    break

    try_merge_improvement()
    
    # Return best found
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
        
    return result
