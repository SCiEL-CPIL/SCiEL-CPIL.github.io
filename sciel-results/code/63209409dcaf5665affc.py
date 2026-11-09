import time
import sys
import random
import copy

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _greedy_partition(n, adj_orig, seed_vertex=None, rng=None):
    """Greedy biclique partition with optional randomization."""
    if rng is None:
        rng = random.Random()
    
    adj = list(adj_orig)  # copy
    bicliques = []
    
    # Find max degree vertex (or use seed)
    if seed_vertex is not None and adj[seed_vertex]:
        u = seed_vertex
    else:
        max_deg = -1
        max_v = -1
        for v in range(n):
            deg = bin(adj[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                max_v = v
        u = max_v
    
    if u == -1:
        return bicliques
    
    # Greedy expansion
    L = [u]
    L_mask = (1 << u)
    R_mask = adj[u] & ~L_mask
    
    # Candidates
    C_L_mask = 0
    for w in range(n):
        if not (L_mask & (1 << w)) and not (R_mask & (1 << w)):
            if (adj[w] & R_mask) == R_mask:
                C_L_mask |= (1 << w)
    
    C_R_mask = 0
    for v in range(n):
        if not (L_mask & (1 << v)) and not (R_mask & (1 << v)):
            if (adj[v] & L_mask) == L_mask:
                C_R_mask |= (1 << v)
    
    # Expand with random tie-breaking
    while True:
        # Find all best candidates
        best_gain = -1
        best_candidates = []
        
        # Candidates for L
        m = C_L_mask
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            gain = bin(R_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_candidates = [(w, 'L')]
            elif gain == best_gain:
                best_candidates.append((w, 'L'))
            m ^= lsb
        
        # Candidates for R
        m = C_R_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            gain = bin(L_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_candidates = [(v, 'R')]
            elif gain == best_gain:
                best_candidates.append((v, 'R'))
            m ^= lsb
        
        if not best_candidates:
            break
        
        # Random choice among best
        choice = rng.choice(best_candidates)
        
        if choice[1] == 'L':
            w = choice[0]
            L.append(w)
            L_mask |= (1 << w)
            C_R_mask &= adj[w]
            C_L_mask &= ~(1 << w)
        else:
            v = choice[0]
            R_mask |= (1 << v)
            C_L_mask &= adj[v]
            C_R_mask &= ~(1 << v)
    
    L_list = [i for i in range(n) if (L_mask & (1 << i))]
    R_list = [i for i in range(n) if (R_mask & (1 << i))]
    
    if not L_list or not R_list:
        # Fallback: single edge
        for u2 in range(n):
            m = adj[u2]
            while m:
                lsb = m & -m
                v2 = lsb.bit_length() - 1
                if u2 < v2:
                    L_list = [u2]
                    R_list = [v2]
                    break
                m ^= lsb
            if L_list:
                break
        if not L_list:
            break
    
    bicliques.append([L_list, R_list])
    
    # Remove edges between L and R
    for u2 in L_list:
        for v2 in R_list:
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)
    
    # Continue until no edges remain
    while True:
        # Check if edges remain
        any_edge = False
        for v in range(n):
            if adj[v]:
                any_edge = True
                break
        if not any_edge:
            break
        
        # Find max degree
        max_deg = -1
        max_v = -1
        for v in range(n):
            deg = bin(adj[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                max_v = v
        
        if max_v == -1:
            break
        
        # Greedy expansion
        u = max_v
        L = [u]
        L_mask = (1 << u)
        R_mask = adj[u] & ~L_mask
        
        C_L_mask = 0
        for w in range(n):
            if not (L_mask & (1 << w)) and not (R_mask & (1 << w)):
                if (adj[w] & R_mask) == R_mask:
                    C_L_mask |= (1 << w)
        
        C_R_mask = 0
        for v in range(n):
            if not (L_mask & (1 << v)) and not (R_mask & (1 << v)):
                if (adj[v] & L_mask) == L_mask:
                    C_R_mask |= (1 << v)
        
        while True:
            best_gain = -1
            best_candidates = []
            
            m = C_L_mask
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                gain = bin(R_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [(w, 'L')]
                elif gain == best_gain:
                    best_candidates.append((w, 'L'))
                m ^= lsb
            
            m = C_R_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                gain = bin(L_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [(v, 'R')]
                elif gain == best_gain:
                    best_candidates.append((v, 'R'))
                m ^= lsb
            
            if not best_candidates:
                break
            
            choice = rng.choice(best_candidates)
            
            if choice[1] == 'L':
                w = choice[0]
                L.append(w)
                L_mask |= (1 << w)
                C_R_mask &= adj[w]
                C_L_mask &= ~(1 << w)
            else:
                v = choice[0]
                R_mask |= (1 << v)
                C_L_mask &= adj[v]
                C_R_mask &= ~(1 << v)
        
        L_list = [i for i in range(n) if (L_mask & (1 << i))]
        R_list = [i for i in range(n) if (R_mask & (1 << i))]
        
        if not L_list or not R_list:
            for u2 in range(n):
                m = adj[u2]
                while m:
                    lsb = m & -m
                    v2 = lsb.bit_length() - 1
                    if u2 < v2:
                        L_list = [u2]
                        R_list = [v2]
                        break
                    m ^= lsb
                if L_list:
                    break
            if not L_list:
                break
        
        bicliques.append([L_list, R_list])
        
        for u2 in L_list:
            for v2 in R_list:
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
    
    return bicliques

def _validate_partition(n, edges, bicliques):
    """Check if bicliques form a valid partition."""
    # Build set of all edges
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # Check each biclique
    covered = set()
    for L, R in bicliques:
        if not L or not R:
            return False
        L_set = set(L)
        R_set = set(R)
        if L_set & R_set:
            return False
        for u in L:
            for v in R:
                if u > v:
                    u, v = v, u
                if (u, v) not in edge_set:
                    return False
                if (u, v) in covered:
                    return False
                covered.add((u, v))
    
    if covered != edge_set:
        return False
    
    return True

def _try_merge(bicliques, i, j, n, edges):
    """Try to merge bicliques i and j. Return new list if successful."""
    L1, R1 = bicliques[i]
    L2, R2 = bicliques[j]
    
    # Option 1: L = L1 ∪ L2, R = R1 ∩ R2
    L = list(set(L1) | set(L2))
    R = list(set(R1) & set(R2))
    if L and R and not (set(L) & set(R)):
        # Check all cross pairs are edges
        valid = True
        for u in L:
            for v in R:
                if (u, v) not in edges_set_global and (v, u) not in edges_set_global:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
            new_bicliques.append([L, R])
            return new_bicliques
    
    # Option 2: L = L1 ∩ L2, R = R1 ∪ R2
    L = list(set(L1) & set(L2))
    R = list(set(R1) | set(R2))
    if L and R and not (set(L) & set(R)):
        valid = True
        for u in L:
            for v in R:
                if (u, v) not in edges_set_global and (v, u) not in edges_set_global:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
            new_bicliques.append([L, R])
            return new_bicliques
    
    # Option 3: L = L1 ∪ L2, R = R1
    L = list(set(L1) | set(L2))
    R = list(R1)
    if L and R and not (set(L) & set(R)):
        valid = True
        for u in L:
            for v in R:
                if (u, v) not in edges_set_global and (v, u) not in edges_set_global:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            # Need to handle remaining edges from biclique j
            # This is complex; skip for now
            pass
    
    # Option 4: L = L1, R = R1 ∪ R2
    L = list(L1)
    R = list(set(R1) | set(R2))
    if L and R and not (set(L) & set(R)):
        valid = True
        for u in L:
            for v in R:
                if (u, v) not in edges_set_global and (v, u) not in edges_set_global:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            pass
    
    return None

# Global edge set for quick lookup
edges_set_global = set()

def _local_search(bicliques, n, edges, max_iters=50, rng=None):
    """Local search: try to merge pairs of bicliques."""
    global edges_set_global
    edges_set_global = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edges_set_global.add((u, v))
    
    if rng is None:
        rng = random.Random()
    
    best = bicliques
    best_len = len(bicliques)
    
    for iteration in range(max_iters):
        improved = False
        # Try all pairs
        for i in range(len(bicliques)):
            for j in range(i + 1, len(bicliques)):
                merged = _try_merge(bicliques, i, j, n, edges)
                if merged is not None and len(merged) < len(bicliques):
                    bicliques = merged
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return bicliques

def construct():
    n = P['n']
    edges = P['edges']
    adj_orig = _build_adj(n, edges)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_bicliques = None
    best_count = float('inf')
    
    # Multi-start
    rng = random.Random(42)
    
    # First: deterministic greedy
    bicliques = _greedy_partition(n, adj_orig, seed_vertex=None, rng=rng)
    if _validate_partition(n, edges, bicliques):
        best_bicliques = bicliques
        best_count = len(bicliques)
    
    # Local search on best
    if best_bicliques and time.time() < deadline:
        refined = _local_search(best_bicliques, n, edges, max_iters=30, rng=rng)
        if _validate_partition(n, edges, refined):
            if len(refined) < best_count:
                best_bicliques = refined
                best_count = len(refined)
    
    # Multiple random restarts
    attempt = 0
    while time.time() < deadline - 5:
        attempt += 1
        # Random seed vertex
        active_vertices = [v for v in range(n) if adj_orig[v]]
        if not active_vertices:
            break
        seed_v = rng.choice(active_vertices)
        
        bicliques = _greedy_partition(n, adj_orig, seed_vertex=seed_v, rng=rng)
        if not _validate_partition(n, edges, bicliques):
            continue
        
        count = len(bicliques)
        if count < best_count:
            best_bicliques = bicliques
            best_count = count
        
        # Local search
        if time.time() < deadline - 3:
            refined = _local_search(bicliques, n, edges, max_iters=20, rng=rng)
            if _validate_partition(n, edges, refined):
                if len(refined) < best_count:
                    best_bicliques = refined
                    best_count = len(refined)
        
        if best_count <= 8:
            break
    
    if best_bicliques is None:
        # Fallback: star decomposition
        bicliques = []
        for u, v in edges:
            bicliques.append([[u], [v]])
        best_bicliques = bicliques
    
    # Convert to required format
    final_result = []
    for bic in best_bicliques:
        l = [int(x) for x in bic[0]]
        r = [int(x) for x in bic[1]]
        final_result.append([l, r])
    
    return final_result
