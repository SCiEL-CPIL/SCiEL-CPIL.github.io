import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    all_mask = (1 << n) - 1

    # Check if a set (bitmask) is a clique
    def is_clique(mask):
        m = mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            m ^= v_bit
            if mask & ~adj[v] & ~(1 << v):
                return False
        return True

    # Check if a set (bitmask) is an independent set
    def is_independent(mask):
        m = mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            m ^= v_bit
            if mask & adj[v]:
                return False
        return True

    # Find all maximal cliques
    def find_maximal_cliques():
        cliques = []
        # Bron-Kerbosch with pivot
        def bronk(R, P_set, X):
            if not P_set and not X:
                if R:
                    cliques.append(R)
                return
            # Choose pivot
            pivot_candidates = P_set | X
            best_pivot = -1
            best_count = -1
            m = pivot_candidates
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                m ^= v_bit
                c = (adj[v] & P_set).bit_count()
                if c > best_count:
                    best_count = c
                    best_pivot = v
            if best_pivot >= 0:
                candidates = P_set & ~adj[best_pivot]
            else:
                candidates = P_set
            m = candidates
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                m ^= v_bit
                bronk(R | v_bit, P_set & adj[v], X & adj[v])
                P_set &= ~v_bit
                X |= v_bit
        bronk(0, all_mask, 0)
        return cliques

    # Find all maximal independent sets (maximal cliques in complement)
    def find_maximal_independent_sets():
        # Complement adjacency
        comp_adj = [0] * n
        for v in range(n):
            comp_adj[v] = all_mask & ~(adj[v] | (1 << v))
        
        indep_sets = []
        def bronk(R, P_set, X):
            if not P_set and not X:
                if R:
                    indep_sets.append(R)
                return
            pivot_candidates = P_set | X
            best_pivot = -1
            best_count = -1
            m = pivot_candidates
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                m ^= v_bit
                c = (comp_adj[v] & P_set).bit_count()
                if c > best_count:
                    best_count = c
                    best_pivot = v
            if best_pivot >= 0:
                candidates = P_set & ~comp_adj[best_pivot]
            else:
                candidates = P_set
            m = candidates
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                m ^= v_bit
                bronk(R | v_bit, P_set & comp_adj[v], X & comp_adj[v])
                P_set &= ~v_bit
                X |= v_bit
        bronk(0, all_mask, 0)
        return indep_sets

    max_cliques = find_maximal_cliques()
    max_indep = find_maximal_independent_sets()

    # Combine all candidate sets (cliques and independent sets)
    # Each candidate is (mask, type) where type 0=clique, 1=independent
    candidates = []
    for c in max_cliques:
        candidates.append((c, 0))
    for s in max_indep:
        candidates.append((s, 1))

    # Remove duplicates: if a set is both a clique and independent (size 1 or 2 with no edge), keep one
    # Actually, keep all since they represent different "types" but same mask - we can dedupe by mask
    # But a singleton is both. Let's just keep all and let the search handle it.
    # Sort by size descending for better pruning
    candidates.sort(key=lambda x: -x[0].bit_count())

    # Deduplicate: if same mask appears with both types, keep both (they're valid as either)
    # Actually for the cover problem, we just need the mask to be valid (clique OR independent)
    # So we can just use unique masks that are either clique or independent
    unique_masks = {}
    for mask, typ in candidates:
        if mask not in unique_masks:
            unique_masks[mask] = typ
    # Use unique masks
    candidate_masks = list(unique_masks.keys())
    candidate_masks.sort(key=lambda m: -m.bit_count())

    # Precompute for each candidate mask, the vertices it covers
    # And for pruning: minimum number of sets needed to cover remaining vertices
    # Use a greedy lower bound: for remaining vertices, find max coverage per set
    
    # Branch and bound
    best_count = n  # Upper bound: each vertex its own part
    best_partition = None

    # Order vertices by degree for heuristic
    order = sorted(range(n), key=lambda v: -adj[v].bit_count())
    # Vertex order index for tie-breaking in search
    vertex_order = {v: i for i, v in enumerate(order)}

    # Precompute: for each candidate mask, the list of vertices
    mask_vertices = {m: [i for i in range(n) if m & (1 << i)] for m in candidate_masks}

    # Lower bound function: minimum sets needed to cover 'remaining' mask
    # Greedy: repeatedly pick the candidate that covers the most remaining vertices
    def lower_bound(remaining):
        if not remaining:
            return 0
        lb = 0
        rem = remaining
        while rem:
            best_cover = 0
            for m in candidate_masks:
                cover = (m & rem).bit_count()
                if cover > best_cover:
                    best_cover = cover
                if time.time() > deadline:
                    return lb + 1  # Conservative
            if best_cover == 0:
                return lb + 1  # Can't cover, but shouldn't happen
            # Remove the best covered vertices (approximate)
            # For a tighter bound, we'd need set cover LP, but greedy is fast
            # Actually for lower bound, we want a lower bound on number of sets
            # Greedy gives an upper bound on the number of sets, not lower
            # Better lower bound: for each remaining vertex, it must be in some set
            # A simple lower bound: ceil(remaining_count / max_set_size_intersecting_remaining)
            # Or: number of remaining vertices / max possible coverage
            # Let's use: for each vertex in remaining, at least one set must cover it
            # If sets can overlap, we need at least 1 per vertex in worst case
            # Better: find the maximum number of remaining vertices any single candidate can cover
            max_cover = 0
            for m in candidate_masks:
                c = (m & rem).bit_count()
                if c > max_cover:
                    max_cover = c
            if max_cover == 0:
                return lb + 1
            lb += 1
            # Remove the vertices covered by the best set (approximate greedy)
            # For a true lower bound, we should not remove; but this is a heuristic
            # Let's use a simpler lower bound: count of remaining / max_cover
            # Actually, let's just return a weak lower bound to avoid complexity
            break
        # Simple lower bound: at least 1 if remaining is non-empty
        return max(lb, 1) if rem else lb

    # Better lower bound: use the fact that each set covers at most max_size vertices
    # Lower bound = ceil(popcount(remaining) / max_candidate_size_intersecting_remaining)
    def lower_bound_simple(remaining):
        if not remaining:
            return 0
        max_cover = 0
        for m in candidate_masks:
            c = (m & remaining).bit_count()
            if c > max_cover:
                max_cover = c
        if max_cover == 0:
            return 1
        import math
        return math.ceil(remaining.bit_count() / max_cover)

    # Branch and bound search
    # State: current set of chosen candidates, current covered mask, current count
    # We search for minimum number of candidates whose union is all_mask and are vertex-disjoint
    
    # Actually, the problem requires vertex-disjoint sets that cover all vertices.
    # So we need a partition, not just a cover.
    
    # Let's use a different approach: recursive search assigning vertices to parts
    # But the directive says to use maximal sets.
    
    # Revised approach: 
    # 1. Find all maximal cliques and maximal independent sets
    # 2. Find minimum number of these (vertex-disjoint) that cover all vertices
    # This is a set partition problem with specific allowed sets
    
    # Use branch and bound on the sets
    # State: (remaining_vertices_mask, chosen_sets, current_count)
    # Try adding each candidate set that is a subset of remaining_vertices
    
    def search(remaining, chosen, count):
        nonlocal best_count, best_partition
        if time.time() > deadline:
            return
        if not remaining:
            if count < best_count:
                best_count = count
                best_partition = list(chosen)
            return
        
        # Pruning: if count >= best_count, no need to continue
        if count >= best_count:
            return
        
        # Lower bound: at least 1 more set needed
        lb = lower_bound_simple(remaining)
        if count + lb >= best_count:
            return
        
        # Try each candidate that is a subset of remaining
        # Order candidates by size descending
        for m in candidate_masks:
            if m & ~remaining:
                continue
            # Check if m is valid (it is, by construction)
            # Recurse
            new_remaining = remaining & ~m
            chosen.append(m)
            search(new_remaining, chosen, count + 1)
            chosen.pop()
            
            # Symmetry breaking: if we've tried a candidate of the same size/type, skip others?
            # Not straightforward. Let's just proceed.
            
            if time.time() > deadline:
                return
        
    search(all_mask, [], 0)
    
    if best_partition is not None:
        # Convert mask list to list of vertex lists
        result = []
        for m in best_partition:
            result.append([i for i in range(n) if m & (1 << i)])
        return result
    
    # Fallback: use the original approach if branch and bound didn't find anything
    # Try k from 1 to n with backtracking
    def solve_k(k, types):
        part_masks = [0] * k
        part_vertices = [[] for _ in range(k)]
        
        def can_add(part_idx, v):
            if types[part_idx] == 0:
                if part_masks[part_idx] & ~adj[v]:
                    return False
                return True
            else:
                if part_masks[part_idx] & adj[v]:
                    return False
            return True

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            seen_empty_types = set()
            
            for i in range(k):
                if not part_vertices[i]:
                    if types[i] in seen_empty_types:
                        continue
                    seen_empty_types.add(types[i])
                
                if can_add(i, v):
                    part_masks[i] |= (1 << v)
                    part_vertices[i].append(v)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    part_masks[i] &= ~(1 << v)
                    part_vertices[i].pop()
                    
            return False
        
        if backtrack(0):
            return [list(p) for p in part_vertices if p]
        return None

    from itertools import product
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        for types in product([0, 1], repeat=k):
            if time.time() > deadline:
                break
            res = solve_k(k, types)
            if res:
                return res

    # Ultimate fallback
    return [[v] for v in range(n)]
