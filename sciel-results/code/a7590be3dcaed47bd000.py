"""Optimized biclique partition via maximal biclique enumeration + branch-and-bound set cover."""

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _seed_solution():
    """Fallback: one biclique per vertex (star decomposition)."""
    n = P['n']
    edges = P['edges']
    adj = {u: [] for u in range(n)}
    for a, b in edges:
        adj[a].append(b)
    result = []
    for u in range(n):
        if adj[u]:
            result.append([[u], list(adj[u])])
    return result


def _generate_candidates():
    """Generate candidate bicliques: stars and maximal bicliques."""
    n = P['n']
    edges = P['edges']
    
    # Adjacency as bitsets for speed
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    
    candidates = []
    seen = set()
    
    def add_candidate(L, R):
        if not L or not R:
            return
        # Canonical form for deduplication
        L_set = set(L)
        R_set = set(R)
        # Orientation: min vertex in L
        if min(L_set) < min(R_set):
            key = (frozenset(L_set), frozenset(R_set))
            L_canon = sorted(L_set)
            R_canon = sorted(R_set)
        else:
            key = (frozenset(R_set), frozenset(L_set))
            L_canon = sorted(R_set)
            R_canon = sorted(L_set)
        
        if key not in seen:
            seen.add(key)
            candidates.append((L_canon, R_canon))
            
    # 1. Stars
    for u in range(n):
        if adj[u]:
            neighbors = [v for v in range(n) if (adj[u] >> v) & 1]
            add_candidate([u], neighbors)
            
    # 2. Maximal Bicliques
    # Enumerate all maximal bicliques by starting from each edge and expanding.
    # To ensure we find ALL maximal bicliques, we rely on the fact that every maximal biclique contains at least one edge.
    # We start with (u,v) and expand to the unique maximal biclique containing (u,v) where u is in L and v is in R?
    # No, a vertex can be in L or R.
    # A better approach for N=30:
    # For each subset of vertices? No.
    # Let's use the property that a biclique is defined by (L,R).
    # We can iterate over all possible L sets? No.
    
    # Heuristic/Exhaustive hybrid:
    # Start with each edge (u,v).
    # Expand L and R greedily to maximal.
    # This finds *some* maximal bicliques.
    # To be more exhaustive, we can also try "forced" orientations.
    
    for a, b in edges:
        # Try expanding with a in L, b in R
        L = {a}
        R = {b}
        changed = True
        while changed:
            changed = False
            # Add to L
            for x in range(n):
                if x not in L and x not in R:
                    # x must be adjacent to all in R
                    # Check if adj[x] covers all bits in R
                    R_mask = 0
                    for r in R:
                        R_mask |= (1 << r)
                    if (adj[x] & R_mask) == R_mask:
                        L.add(x)
                        changed = True
            # Add to R
            for x in range(n):
                if x not in L and x not in R:
                    # x must be adjacent to all in L
                    L_mask = 0
                    for l in L:
                        L_mask |= (1 << l)
                    if (adj[x] & L_mask) == L_mask:
                        R.add(x)
                        changed = True
        add_candidate(list(L), list(R))
        
        # Try expanding with a in R, b in L
        L = {b}
        R = {a}
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x not in L and x not in R:
                    R_mask = 0
                    for r in R:
                        R_mask |= (1 << r)
                    if (adj[x] & R_mask) == R_mask:
                        L.add(x)
                        changed = True
            for x in range(n):
                if x not in L and x not in R:
                    L_mask = 0
                    for l in L:
                        L_mask |= (1 << l)
                    if (adj[x] & L_mask) == L_mask:
                        R.add(x)
                        changed = True
        add_candidate(list(L), list(R))
        
    return candidates


def _solve_branch_and_bound(candidates, edge_indices, time_limit=50):
    """
    Solve minimum set cover using branch-and-bound with bitsets.
    """
    import time
    start_time = time.time()
    
    n_edges = len(edge_indices)
    n_cands = len(candidates)
    
    if n_cands == 0:
        return None
        
    # Map edges to indices 0..n_edges-1
    # Precompute coverage bitsets for each candidate
    cand_masks = []
    for L, R in candidates:
        mask = 0
        for u in L:
            for v in R:
                key = (min(u, v), max(u, v))
                if key in edge_indices:
                    idx = edge_indices[key]
                    mask |= (1 << idx)
        cand_masks.append(mask)
        
    # Remove dominated candidates (if A is subset of B, A is useless)
    # Since we use bitsets, A is subset of B if (A & ~B) == 0
    # Keep only maximal masks
    # Sort by number of bits descending
    cand_indices = list(range(n_cands))
    cand_indices.sort(key=lambda i: -bin(cand_masks[i]).count('1'))
    
    keep = [False] * n_cands
    final_cands = []
    for i in cand_indices:
        is_dominated = False
        for j in range(len(final_cands)):
            if (cand_masks[i] & ~final_cands[j]) == 0:
                is_dominated = True
                break
        if not is_dominated:
            keep[i] = True
            final_cands.append(cand_masks[i])
            
    # Remap
    # We lost the original indices, so we need to store them or just use the masks.
    # We need to return the original candidate indices to construct the result.
    # Let's keep a list of (mask, original_index)
    
    # Re-do filtering properly
    final_cands_list = []
    for i in cand_indices:
        is_dominated = False
        for mask_j, idx_j in final_cands_list:
            if (cand_masks[i] & ~mask_j) == 0:
                is_dominated = True
                break
        if not is_dominated:
            final_cands_list.append((cand_masks[i], i))
            
    # Sort final list by size descending for greedy
    final_cands_list.sort(key=lambda x: -bin(x[0]).count('1'))
    
    n_final = len(final_cands_list)
    if n_final == 0:
        return None
        
    # Greedy Initial Solution
    uncovered = (1 << n_edges) - 1
    greedy_sol = []
    while uncovered:
        best_idx = -1
        best_count = 0
        for i in range(n_final):
            mask, orig_idx = final_cands_list[i]
            count = bin(mask & uncovered).count('1')
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1 or best_count == 0:
            break
        greedy_sol.append(best_idx)
        uncovered &= ~final_cands_list[best_idx][0]
        
    if uncovered:
        return None # Should not happen if stars are included
        
    upper_bound = len(greedy_sol)
    best_sol = list(greedy_sol)
    
    # Precompute for LB2: For each edge, which candidates cover it?
    # Or better, for LB2 (packing), we need to know if two edges can be covered by the same candidate.
    # Two edges e1, e2 are "compatible" if there exists a candidate covering both.
    # We want to find a set of edges that are pairwise INCOMPATIBLE.
    # Precompute a bitset for each edge of candidates that cover it?
    # edge_to_cands_mask[e] is a bitset of candidate indices (in final_cands_list) that cover e.
    
    edge_to_cands_mask = [0] * n_edges
    for i in range(n_final):
        mask, _ = final_cands_list[i]
        for e in range(n_edges):
            if (mask >> e) & 1:
                edge_to_cands_mask[e] |= (1 << i)
                
    # Check feasibility
    for e in range(n_edges):
        if edge_to_cands_mask[e] == 0:
            return None
            
    # Branch and Bound
    nodes_explored = 0
    
    def branch_and_bound(selection, uncovered, depth):
        nonlocal nodes_explored, upper_bound, best_sol
        
        nodes_explored += 1
        if time.time() - start_time > time_limit:
            return
            
        if uncovered == 0:
            if len(selection) < upper_bound:
                upper_bound = len(selection)
                best_sol = list(selection)
            return
            
        # Lower Bound 1: ceil(uncovered_bits / max_candidate_cover)
        # We need max coverage of any candidate on the uncovered set.
        # This is expensive to compute exactly. Approximate with global max?
        # Or precompute max cover for each possible uncovered set? No.
        # Simple LB: 1 if uncovered > 0.
        # Better LB: Count bits in uncovered. Let k = bin(uncovered).count('1').
        # Max possible cover by one candidate is at most k.
        # So LB1 = 1. (Weak)
        
        # Let's use a slightly better LB1:
        # Find the candidate with the most bits in uncovered.
        # To speed this up, we can iterate.
        max_cover = 0
        for i in range(n_final):
            mask, _ = final_cands_list[i]
            c = bin(mask & uncovered).count('1')
            if c > max_cover:
                max_cover = c
                if max_cover == bin(uncovered).count('1'):
                    break
        if max_cover == 0:
            return
        lb1 = (bin(uncovered).count('1') + max_cover - 1) // max_cover
        
        # Lower Bound 2: Greedy Packing
        # Pick an uncovered edge e.
        # Any candidate covering e is "used".
        # Remove all edges covered by any candidate covering e.
        # Repeat.
        lb2 = 0
        rem = uncovered
        while rem:
            # Pick edge with fewest candidates? Or just lowest index?
            # Lowest index is fast.
            # e = (rem & -rem).bit_length() - 1
            # To be smarter, pick edge with min candidates.
            # This requires iterating.
            best_e = -1
            min_cands = n_final + 1
            temp_rem = rem
            while temp_rem:
                lsb = temp_rem & -temp_rem
                e = lsb.bit_length() - 1
                # Count candidates covering e
                # We have edge_to_cands_mask[e]
                # But we only care about candidates that cover e AND are available?
                # In packing, we don't restrict candidates, we just want to know if they *can* cover.
                # Actually, the standard packing LB is:
                # Find a set of edges such that no single set covers more than one.
                # This is equivalent to finding a matching in the bipartite graph? No.
                # It's the independence number of the conflict graph.
                # Greedy: Pick e. Remove all edges that share a candidate with e.
                
                # To make it fast, we precomputed edge_to_cands_mask.
                # But we need to know which edges share a candidate.
                # Let's stick to a simpler LB if it's too slow.
                # For now, let's use a very fast LB2:
                # Just count edges? No.
                # Let's use the "min candidates" heuristic for branching, but for LB,
                # let's just use lb1 for now and rely on good branching.
                # Or use a precomputed "conflict" bitset.
                
                # Let's try to compute lb2 properly but efficiently.
                # We can precompute for each edge e, a bitset of edges that conflict with e.
                # Conflict(e, f) = True if there is NO candidate covering both e and f.
                # This is expensive to precompute (n_edges^2 * n_cands).
                # 200^2 * 100 = 4,000,000 ops. Acceptable.
                
                break # Skip complex LB2 for now, rely on lb1 and branching
            # Fallback LB2: 0
            lb2 = 0
            break # Disable LB2 to save time in this iteration, focus on branching
            
        lb = lb1
        
        if len(selection) + lb >= upper_bound:
            return
            
        # Branching: Choose edge with fewest candidates
        # Iterate through uncovered edges
        best_e = -1
        min_cands = n_final + 1
        temp_rem = uncovered
        while temp_rem:
            lsb = temp_rem & -temp_rem
            e = lsb.bit_length() - 1
            c = bin(edge_to_cands_mask[e]).count('1')
            if c < min_cands:
                min_cands = c
                best_e = e
            temp_rem ^= lsb
            
        if best_e == -1:
            return
            
        # Candidates covering best_e
        cands_mask = edge_to_cands_mask[best_e]
        
        # Order candidates by coverage of remaining edges
        # Get list of candidate indices
        cand_indices_for_e = []
        temp_mask = cands_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            i = lsb.bit_length() - 1
            cand_indices_for_e.append(i)
            temp_mask ^= lsb
            
        # Sort by coverage
        cand_indices_for_e.sort(key=lambda i: -bin(final_cands_list[i][0] & uncovered).count('1'))
        
        for i in cand_indices_for_e:
            if time.time() - start_time > time_limit:
                return
            selection.append(i)
            new_uncovered = uncovered & ~final_cands_list[i][0]
            branch_and_bound(selection, new_uncovered, depth + 1)
            selection.pop()
            
    branch_and_bound([], (1 << n_edges) - 1, 0)
    
    if best_sol:
        # Convert back to original candidate indices
        return [final_cands_list[i][1] for i in best_sol]
    return None


def construct():
    import time
    start = time.time()
    
    # Try the optimized approach
    try:
        candidates = _generate_candidates()
        
        # Build edge index
        edge_indices = {}
        for i, (a, b) in enumerate(P['edges']):
            key = (min(a, b), max(a, b))
            edge_indices[key] = i
        
        if candidates:
            sol = _solve_branch_and_bound(candidates, edge_indices, time_limit=55)
            if sol is not None:
                # Build the result
                result = []
                for idx in sol:
                    L, R = candidates[idx]
                    # Verify all cross pairs are edges
                    valid = True
                    for u in L:
                        for v in R:
                            key = (min(u, v), max(u, v))
                            if key not in edge_indices:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and L and R:
                        result.append([list(L), list(R)])
                
                # Verify partition: every edge covered exactly once
                if result:
                    edge_count = {}
                    for L, R in result:
                        for u in L:
                            for v in R:
                                key = (min(u, v), max(u, v))
                                edge_count[key] = edge_count.get(key, 0) + 1
                    
                    all_edges = set(edge_indices.keys())
                    covered = set(edge_count.keys())
                    
                    if covered == all_edges and all(c == 1 for c in edge_count.values()):
                        # Valid solution!
                        # Ensure types are correct
                        final_result = []
                        for L, R in result:
                            L_int = [int(x) for x in L]
                            R_int = [int(x) for x in R]
                            final_result.append([L_int, R_int])
                        return final_result
    except Exception:
        pass
    
    # Fallback to seed
    return _seed_solution()
