"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        u, v = e
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u in range(n):
        for v in range(n):
            if (u, v) in edge_set:
                adj[u] |= (1 << v)
    
    # Precompute: for each vertex, its non-neighbors (for independent sets)
    # non_adj[u] has bits set for vertices not adjacent to u (and u != v)
    non_adj = [0] * n
    for u in range(n):
        mask = 0
        for v in range(n):
            if u != v and (u, v) not in edge_set:
                mask |= (1 << v)
        non_adj[u] = mask
    
    def is_clique(vertices):
        """Check if vertices form a clique using bitmasks."""
        if not vertices:
            return True
        v0 = vertices[0]
        for v in vertices[1:]:
            if not (adj[v0] & (1 << v)):
                return False
        return True
    
    def is_independent(vertices):
        """Check if vertices form an independent set using bitmasks."""
        if not vertices:
            return True
        v0 = vertices[0]
        for v in vertices[1:]:
            if (adj[v0] & (1 << v)):
                return False
        return True
    
    # Generate all maximal cliques and maximal independent sets
    # We'll generate all cliques and independent sets, then filter to maximal ones
    # For n=14, this is feasible but we need to be careful
    
    all_cliques = []
    all_independent = []
    
    # Generate all subsets and check if they are cliques/independent
    # For n=14, there are 2^14 = 16384 subsets, which is manageable
    
    # Actually, let's be smarter: generate all maximal cliques using Bron-Kerbosch
    # and all maximal independent sets (which are maximal cliques in the complement)
    
    def bron_kerbosch(R, P_set, X, adj_mask):
        """Bron-Kerbosch algorithm to find all maximal cliques."""
        if not P_set and not X:
            if R:
                all_cliques.append(frozenset(R))
            return
        # Choose pivot to reduce recursion
        if P_set:
            pivot = next(iter(P_set))
            candidates = P_set - {v for v in P_set if (adj_mask[v] & (1 << pivot))}
        else:
            candidates = set()
        
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & {u for u in P_set if (adj_mask[u] & (1 << v))}, 
                         X & {u for u in X if (adj_mask[u] & (1 << v))}, adj_mask)
            P_set.remove(v)
            X.add(v)
    
    # Find maximal cliques
    bron_kerbosch(set(), set(range(n)), set(), adj)
    
    # Find maximal independent sets (maximal cliques in complement)
    comp_adj = [0] * n
    for u in range(n):
        for v in range(n):
            if u != v and (u, v) not in edge_set:
                comp_adj[u] |= (1 << v)
    
    all_independent.clear()
    bron_kerbosch(set(), set(range(n)), set(), comp_adj)
    
    # Combine all homogeneous sets (cliques and independent sets)
    homogeneous_sets = []
    seen = set()
    for cs in all_cliques:
        if cs not in seen:
            homogeneous_sets.append(cs)
            seen.add(cs)
    for iset in all_independent:
        if iset not in seen:
            homogeneous_sets.append(iset)
            seen.add(iset)
    
    # Filter: keep only sets that are actually cliques or independent sets
    # (Bron-Kerbosch should give maximal ones, but let's verify)
    valid_sets = []
    for s in homogeneous_sets:
        verts = list(s)
        if is_clique(verts) or is_independent(verts):
            valid_sets.append(s)
    
    # Now solve minimum set cover: find minimum number of disjoint homogeneous sets covering all vertices
    # Use branch and bound
    
    all_vertices_mask = (1 << n) - 1
    best_partition = None
    best_count = n + 1
    
    # Sort sets by size descending for better pruning
    valid_sets_sorted = sorted(valid_sets, key=lambda s: -len(s))
    
    # Precompute bitmask for each set
    set_masks = []
    for s in valid_sets_sorted:
        mask = 0
        for v in s:
            mask |= (1 << v)
        set_masks.append(mask)
    
    def solve(remaining_mask, current_partition, current_count):
        nonlocal best_partition, best_count
        
        if current_count >= best_count:
            return
        
        if remaining_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_partition = [set(p) for p in current_partition]
            return
        
        # Lower bound: at least ceil(popcount(remaining) / max_set_size) more sets needed
        popcount = bin(remaining_mask).count('1')
        max_possible = 1
        for mask in set_masks:
            if mask & remaining_mask == mask:
                sz = bin(mask).count('1')
                if sz > max_possible:
                    max_possible = sz
        if max_possible == 0:
            return
        lb = (popcount + max_possible - 1) // max_possible
        if current_count + lb >= best_count:
            return
        
        # Try each valid set that can be added
        for i, mask in enumerate(set_masks):
            if mask & remaining_mask == mask:  # set is subset of remaining
                new_remaining = remaining_mask & ~mask
                current_partition.append(valid_sets_sorted[i])
                solve(new_remaining, current_partition, current_count + 1)
                current_partition.pop()
                
                # Prune: if we found optimal (1 set), stop
                if best_count == 1:
                    return
    
    solve(all_vertices_mask, [], 0)
    
    # Convert best_partition to list of lists
    if best_partition is None:
        # Fallback: each vertex is its own part
        return [[v] for v in range(n)]
    
    result = []
    for s in best_partition:
        result.append(sorted(list(s)))
    
    return result

# EVOLVE-BLOCK-END
