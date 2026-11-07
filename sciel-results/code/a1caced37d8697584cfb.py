"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    es = set()
    for e in P['edges']:
        es.add(tuple(e))
        es.add(tuple(reversed(e)))
    
    # Adjacency as sets
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
        
    # Complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)

    deadline = time.time() + 50  # safety margin

    # Precompute is_adj matrix for fast lookup
    is_adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            is_adj[i][j] = True

    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    order_idx = {v: i for i, v in enumerate(order)}

    # Compute chromatic number of G and comp(G) using backtracking
    def chromatic_number(adj_sets):
        n_local = len(adj_sets)
        # Simple backtracking for chromatic number
        # Colors 0..k-1
        # Try k=1..n
        for k in range(1, n_local + 1):
            colors = [-1] * n_local
            def bt(v):
                if v == n_local:
                    return True
                for c in range(k):
                    conflict = False
                    for u in adj_sets[v]:
                        if colors[u] == c:
                            conflict = True
                            break
                    if not conflict:
                        colors[v] = c
                        if bt(v + 1):
                            return True
                        colors[v] = -1
                return False
            if bt(0):
                return k
        return n_local

    chi_g = chromatic_number(adj)
    chi_comp_g = chromatic_number(comp_adj)
    
    # Lower bound on k
    lb = chi_g + chi_comp_g
    if lb < 1:
        lb = 1

    best_parts = None
    best_count = n + 1

    # Precompute for each vertex, which other vertices are adjacent/non-adjacent
    # For fast clique/independent set checks
    
    def can_add_to_clique(part, v):
        for u in part:
            if not is_adj[u][v]:
                return False
        return True

    def can_add_to_independent(part, v):
        for u in part:
            if is_adj[u][v]:
                return False
        return True

    # Backtracking function for a fixed k and c (number of clique parts)
    # Parts 0..c-1 are cliques, parts c..k-1 are independent sets
    def solve_for_k_c(k, c):
        parts = [[] for _ in range(k)]
        assigned = [False] * n
        
        # Heuristic: assign vertices in 'order'
        # To break symmetry, we can enforce that the first vertex in each part is the smallest index?
        # Or just rely on the order.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                return True
            v = order[idx]
            
            # Try adding v to each part
            for p_idx in range(k):
                if p_idx < c:
                    # Clique part
                    if can_add_to_clique(parts[p_idx], v):
                        parts[p_idx].append(v)
                        if backtrack(idx + 1):
                            return True
                        parts[p_idx].pop()
                else:
                    # Independent part
                    if can_add_to_independent(parts[p_idx], v):
                        parts[p_idx].append(v)
                        if backtrack(idx + 1):
                            return True
                        parts[p_idx].pop()
            return False

        if backtrack(0):
            # Collect non-empty parts
            result = []
            for p in parts:
                if p:
                    result.append(list(p))
            return result
        return None

    # Iterate k from lb upwards
    for k in range(lb, n + 1):
        if time.time() > deadline:
            break
        if k >= best_count:
            break
            
        # c is number of clique parts
        # c >= chi_comp_g
        # k - c >= chi_g  => c <= k - chi_g
        min_c = max(0, chi_comp_g)
        max_c = min(k, k - chi_g)
        
        if min_c > max_c:
            continue
            
        for c in range(min_c, max_c + 1):
            if time.time() > deadline:
                break
                
            result = solve_for_k_c(k, c)
            if result is not None:
                # Check if all parts are non-empty? 
                # The problem says "partition of vertices ... into nonempty lists".
                # Our backtracking might produce empty parts. We should filter them.
                # But if we filter, the number of parts might be less than k.
                # However, we are iterating k, so if we find a solution with k parts,
                # and some are empty, it's actually a solution with fewer parts.
                # But we want to minimize the number of parts.
                # So if we find a valid partition with k parts, and some are empty,
                # we should treat it as a partition with fewer parts.
                # But our backtracking assigns all vertices, so if a part is empty,
                # it means we didn't use it.
                # So we can just filter out empty parts.
                
                # However, if we filter, the number of parts might be less than k.
                # This is fine, it's a valid partition with fewer parts.
                # But we are iterating k, so we would have found it earlier if it was valid.
                # Unless the backtracking failed to find a solution with fewer parts.
                # But if a solution with fewer parts exists, the backtracking for that k
                # should have found it (assuming no symmetry issues).
                # To be safe, we can just return the filtered list.
                
                filtered = [p for p in result if p]
                if filtered:
                    best_parts = filtered
                    best_count = len(filtered)
                    # Since we iterate k upwards, the first solution found is optimal?
                    # Not necessarily, because we might find a solution with k parts
                    # that has empty parts, so the actual number is < k.
                    # But if we found a solution with k parts, and it has empty parts,
                    # it's a solution with fewer parts.
                    # So we can break immediately?
                    # Yes, because we want to minimize the number of parts.
                    # If we found a valid partition, and we are iterating k upwards,
                    # the first valid partition we find is the one with the minimum k
                    # that allows a valid partition.
                    # But wait, the number of parts in the result might be less than k.
                    # So we should return it immediately.
                    return [list(p) for p in filtered]
                    
    if best_parts is None:
        # Fallback: each vertex its own part
        best_parts = [[v] for v in range(n)]
        
    # Ensure all parts are non-empty and return as list of lists
    final = []
    for p in best_parts:
        if p:
            final.append(list(p))
            
    return final

# EVOLVE-BLOCK-END
