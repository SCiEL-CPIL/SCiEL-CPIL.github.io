"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_edge_set():
    es = set()
    for u, v in P['edges']:
        es.add((min(u, v), max(u, v)))
    return es

def _build_adj_mask(edge_set):
    n = P['n']
    adj_mask = [0] * n
    for u, v in edge_set:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _mask_to_list(mask, n):
    return [i for i in range(n) if (mask >> i) & 1]

def _find_all_maximal_bicliques(adj_mask, edge_set, deadline):
    """
    Find maximal bicliques using a systematic approach.
    For each edge, try to expand it into a maximal biclique.
    Use bitmasks for efficiency.
    """
    n = P['n']
    all_mask = (1 << n) - 1
    bicliques = []
    seen = set()
    
    # For each edge, expand to maximal biclique
    edges = list(edge_set)
    random.shuffle(edges)
    
    for seed_u, seed_v in edges:
        if time.time() > deadline:
            break
        
        # Start with L={u}, R={v}
        L_mask = (1 << seed_u)
        R_mask = (1 << seed_v)
        
        # Expand to maximal
        changed = True
        while changed:
            changed = False
            # Try adding to L: vertex x not in L∪R, connected to all in R
            candidates_L = all_mask & ~(L_mask | R_mask)
            # For each candidate, check if connected to all in R
            temp = candidates_L
            while temp:
                x_bit = temp & (-temp)
                x = x_bit.bit_length() - 1
                temp -= x_bit
                if (adj_mask[x] & R_mask) == R_mask:
                    L_mask |= x_bit
                    changed = True
            
            # Try adding to R: vertex x not in L∪R, connected to all in L
            candidates_R = all_mask & ~(L_mask | R_mask)
            temp = candidates_R
            while temp:
                x_bit = temp & (-temp)
                x = x_bit.bit_length() - 1
                temp -= x_bit
                if (adj_mask[x] & L_mask) == L_mask:
                    R_mask |= x_bit
                    changed = True
        
        if L_mask and R_mask:
            key = (L_mask, R_mask)
            key2 = (R_mask, L_mask)
            if key not in seen and key2 not in seen:
                seen.add(key)
                seen.add(key2)
                bicliques.append((L_mask, R_mask))
    
    return bicliques

def _biclique_edge_mask(L_mask, R_mask, edge_to_bit, edge_set):
    """Compute the bitmask of edges covered by this biclique."""
    edge_mask = 0
    for u in _mask_to_list(L_mask, P['n']):
        for v in _mask_to_list(R_mask, P['n']):
            e = (min(u, v), max(u, v))
            if e in edge_to_bit:
                edge_mask |= (1 << edge_to_bit[e])
    return edge_mask

def _greedy_partition_bitmask(adj_mask, edge_set, deadline):
    """
    Greedy: repeatedly find the largest biclique in remaining edges, remove those edges.
    Uses bitmasks for speed.
    """
    n = P['n']
    all_mask = (1 << n) - 1
    remaining = set(edge_set)
    bicliques = []
    
    # Build edge index
    edge_list = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    num_edges = len(edge_list)
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Build adjacency for remaining edges
        rem_adj = [0] * n
        rem_verts = set()
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
            rem_verts.add(u)
            rem_verts.add(v)
        
        # Find best biclique in remaining
        best_L = 0
        best_R = 0
        best_size = 0
        
        verts_list = list(rem_verts)
        # Try each vertex as anchor for left side
        for anchor in verts_list:
            if time.time() > deadline:
                break
            
            # Neighbors of anchor in remaining
            R_mask = rem_adj[anchor]
            L_mask = (1 << anchor)
            
            # Expand
            changed = True
            while changed:
                changed = False
                # Add to L
                candidates = rem_verts_mask = 0
                for v in rem_verts:
                    candidates |= (1 << v)
                candidates &= ~(L_mask | R_mask)
                temp = candidates
                while temp:
                    x_bit = temp & (-temp)
                    x = x_bit.bit_length() - 1
                    temp -= x_bit
                    if (rem_adj[x] & R_mask) == R_mask:
                        L_mask |= x_bit
                        changed = True
                
                # Add to R
                candidates = 0
                for v in rem_verts:
                    candidates |= (1 << v)
                candidates &= ~(L_mask | R_mask)
                temp = candidates
                while temp:
                    x_bit = temp & (-temp)
                    x = x_bit.bit_length() - 1
                    temp -= x_bit
                    if (rem_adj[x] & L_mask) == L_mask:
                        R_mask |= x_bit
                        changed = True
            
            size = bin(L_mask).count('1') * bin(R_mask).count('1')
            if size > best_size:
                best_size = size
                best_L = L_mask
                best_R = R_mask
        
        if best_L == 0 or best_R == 0:
            # Fallback: single edge
            u, v = next(iter(remaining))
            bicliques.append(((1 << u), (1 << v)))
            remaining.discard((u, v))
        else:
            bicliques.append((best_L, best_R))
            # Remove covered edges
            for u in _mask_to_list(best_L, n):
                for v in _mask_to_list(best_R, n):
                    remaining.discard((min(u, v), max(u, v)))
    
    return bicliques

def _solve_set_cover(biclique_masks, num_edges, deadline, initial_bound):
    """
    Solve minimum set cover using backtracking with bitmasks.
    biclique_masks: list of int bitmasks, each representing edges covered by that biclique.
    Returns list of indices of selected bicliques.
    """
    full_mask = (1 << num_edges) - 1
    
    # Precompute: for each edge, which bicliques cover it
    edge_to_bicliques = [0] * num_edges
    for i, bm in enumerate(biclique_masks):
        for e in range(num_edges):
            if (bm >> e) & 1:
                edge_to_bicliques[e] |= (1 << i)
    
    num_cand = len(biclique_masks)
    
    # Greedy for initial upper bound
    remaining = full_mask
    chosen = []
    while remaining:
        best_bi = -1
        best_count = 0
        for i in range(num_cand):
            cnt = bin(biclique_masks[i] & remaining).count('1')
            if cnt > best_count:
                best_count = cnt
                best_bi = i
        if best_bi < 0:
            break
        chosen.append(best_bi)
        remaining &= ~biclique_masks[best_bi]
    
    best_solution = list(chosen)
    best_size = len(chosen)
    
    # Backtracking search
    memo = {}
    node_count = [0]
    
    def backtrack(uncovered, current, depth):
        nonlocal best_solution, best_size
        
        node_count[0] += 1
        if time.time() > deadline or node_count[0] > 500000:
            return
        
        if uncovered == 0:
            if depth < best_size:
                best_size = depth
                best_solution = list(current)
            return
        
        if depth >= best_size:
            return
        
        # Lower bound
        max_cov = 0
        for i in range(num_cand):
            cov = bin(biclique_masks[i] & uncovered).count('1')
            if cov > max_cov:
                max_cov = cov
        if max_cov == 0:
            return
        lb = (bin(uncovered).count('1') + max_cov - 1) // max_cov
        if depth + lb >= best_size:
            return
        
        # Memoization
        if uncovered in memo:
            if memo[uncovered] <= depth:
                return
        else:
            memo[uncovered] = depth
        
        # Most constrained edge
        min_count = num_cand + 1
        min_edge = -1
        temp = uncovered
        while temp:
            e_bit = temp & (-temp)
            e = e_bit.bit_length() - 1
            temp -= e_bit
            cnt = bin(edge_to_bicliques[e]).count('1')
            if cnt < min_count:
                min_count = cnt
                min_edge = e
        
        if min_edge < 0 or min_count == 0:
            return
        
        # Try bicliques covering min_edge
        candidates_mask = edge_to_bicliques[min_edge]
        temp_cand = candidates_mask
        # Sort by coverage
        cand_list = []
        while temp_cand:
            bi_bit = temp_cand & (-temp_cand)
            bi = bi_bit.bit_length() - 1
            temp_cand -= bi_bit
            cov = bin(biclique_masks[bi] & uncovered).count('1')
            cand_list.append((cov, bi))
        cand_list.sort(reverse=True)
        
        for cov, bi in cand_list:
            new_uncovered = uncovered & ~biclique_masks[bi]
            current.append(bi)
            backtrack(new_uncovered, current, depth + 1)
            current.pop()
    
    backtrack(full_mask, [], 0)
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    edge_set = _build_edge_set()
    n = P['n']
    num_edges = len(edge_set)
    
    # Build adjacency masks
    adj_mask = _build_adj_mask(edge_set)
    
    # Phase 1: Greedy partition for initial upper bound
    greedy_bicliques = _greedy_partition_bitmask(adj_mask, edge_set, deadline)
    greedy_count = len(greedy_bicliques)
    
    # Phase 2: Find candidate maximal bicliques
    candidates = _find_all_maximal_bicliques(adj_mask, edge_set, deadline)
    
    # Also add the greedy bicliques as candidates
    all_candidates = list(candidates)
    for L, R in greedy_bicliques:
        key = (L, R)
        if key not in [(a, b) for a, b in all_candidates] and (R, L) not in [(a, b) for a, b in all_candidates]:
            all_candidates.append((L, R))
    
    # Build edge-to-index mapping
    edge_list = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    # Convert candidates to edge bitmasks
    biclique_edge_masks = []
    biclique_vertices = []
    seen_edge_masks = set()
    
    for L_mask, R_mask in all_candidates:
        em = 0
        valid = True
        for u in _mask_to_list(L_mask, n):
            for v in _mask_to_list(R_mask, n):
                e = (min(u, v), max(u, v))
                if e in edge_to_idx:
                    em |= (1 << edge_to_idx[e])
                else:
                    valid = False
                    break
            if not valid:
                break
        if valid and em:
            if em not in seen_edge_masks:
                seen_edge_masks.add(em)
                biclique_edge_masks.append(em)
                biclique_vertices.append((L_mask, R_mask))
    
    # Phase 3: Solve set cover
    best_solution = None
    if biclique_edge_masks:
        best_indices = _solve_set_cover(biclique_edge_masks, num_edges, deadline, greedy_count)
        if best_indices:
            best_solution = best_indices
    
    # Build result
    if best_solution and len(best_solution) < greedy_count:
        result_masks = [biclique_vertices[i] for i in best_solution]
    else:
        result_masks = greedy_bicliques
    
    # Convert to proper format
    result = []
    for L_mask, R_mask in result_masks:
        L_list = [int(v) for v in _mask_to_list(L_mask, n)]
        R_list = [int(v) for v in _mask_to_list(R_mask, n)]
        if L_list and R_list:
            result.append([L_list, R_list])
    
    # Validate
    if not result:
        # Fallback: star decomposition
        for u in range(n):
            neighbors = _mask_to_list(adj_mask[u], n)
            if neighbors:
                result.append([neighbors, [u]])
    
    return result
# EVOLVE-BLOCK-END
