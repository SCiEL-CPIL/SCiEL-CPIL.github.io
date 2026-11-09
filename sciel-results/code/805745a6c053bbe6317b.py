"""Rewritten program: exact cochromatic partition via branch-and-bound with bitmasks."""

import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Full mask of all vertices
    full_mask = (1 << n) - 1
    
    # Precompute: for each vertex v, the set of vertices that can be in a clique with v
    # (i.e., adjacent to v) and in an independent set with v (non-adjacent to v, excluding self)
    clique_compat = [0] * n
    indep_compat = [0] * n
    for v in range(n):
        # clique_compat[v]: vertices adjacent to v
        clique_compat[v] = adj[v]
        # indep_compat[v]: vertices NOT adjacent to v, excluding v itself
        non_adj = full_mask ^ (1 << v)
        non_adj &= ~adj[v]
        indep_compat[v] = non_adj
    
    # Greedy upper bound
    def greedy_upper_bound():
        assigned = [False] * n
        parts = []
        for v in range(n):
            if assigned[v]:
                continue
            # Try to add v to an existing part
            placed = False
            for i, part in enumerate(parts):
                # Check if part is clique or independent set
                # Determine type from first two elements
                if len(part) >= 2:
                    a, b = part[0], part[1]
                    is_clique = (1 << b) & adj[a]
                else:
                    is_clique = None  # undetermined
                
                if is_clique is None:
                    # Single element, can be either
                    # Try clique first, then independent
                    # For simplicity, just try to add if compatible
                    # Actually, let's determine: if we add v, check consistency
                    # For a single vertex part, it's both clique and independent
                    # We need to decide the type. Let's try both.
                    # For greedy, just pick one type.
                    # Let's say we try to make it a clique if v is adjacent to part[0]
                    if (1 << v) & adj[part[0]]:
                        # Try as clique
                        if all((1 << v) & adj[u] for u in part):
                            part.append(v)
                            assigned[v] = True
                            placed = True
                            break
                    else:
                        # Try as independent set
                        if all(not ((1 << v) & adj[u]) for u in part):
                            part.append(v)
                            assigned[v] = True
                            placed = True
                            break
                elif is_clique:
                    if all((1 << v) & adj[u] for u in part):
                        part.append(v)
                        assigned[v] = True
                        placed = True
                        break
                else:
                    if all(not ((1 << v) & adj[u]) for u in part):
                        part.append(v)
                        assigned[v] = True
                        placed = True
                        break
            if not placed:
                parts.append([v])
                assigned[v] = True
        return len(parts)
    
    ub = greedy_upper_bound()
    
    # Branch and bound
    best_parts = None
    best_k = ub
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))
    
    # Use a recursive approach with state:
    # - assigned: bitmask of assigned vertices
    # - parts: list of (type, vertex_set_mask)
    #   type: 0 = clique, 1 = independent set, -1 = undetermined (single vertex)
    
    deadline = time.time() + 85.0  # 5s margin before 90s kill
    
    def check_clique(mask, v):
        """Check if v can be added to clique with vertices in mask."""
        m = mask
        while m:
            u = (m & -m).bit_length() - 1
            m &= m - 1
            if not ((1 << v) & adj[u]):
                return False
        return True
    
    def check_indep(mask, v):
        """Check if v can be added to independent set with vertices in mask."""
        m = mask
        while m:
            u = (m & -m).bit_length() - 1
            m &= m - 1
            if (1 << v) & adj[u]:
                return False
        return True
    
    def solve(assigned_mask, parts, depth):
        nonlocal best_parts, best_k
        
        if time.time() > deadline:
            return
        
        # If all vertices assigned
        if assigned_mask == full_mask:
            if len(parts) < best_k:
                best_k = len(parts)
                # Reconstruct parts from masks
                best_parts = [list(range(n)) for _ in range(len(parts))]
                for i, (ptype, pmask) in enumerate(parts):
                    verts = []
                    m = pmask
                    while m:
                        u = (m & -m).bit_length() - 1
                        m &= m - 1
                        verts.append(u)
                    best_parts[i] = verts
            return
        
        # Pruning: if current parts count + lower bound >= best_k
        # Lower bound: number of unassigned vertices / max possible in a part
        # Simple lower bound: at least 1 if any unassigned
        unassigned_count = n - bin(assigned_mask).count('1')
        if len(parts) + 1 > best_k:
            return
        
        # Pick next vertex to assign
        # Use the first unassigned vertex in our ordering
        v = -1
        for candidate in order:
            if not (assigned_mask & (1 << candidate)):
                v = candidate
                break
        
        if v == -1:
            return
        
        v_bit = 1 << v
        
        # Try to add v to each existing part
        for i in range(len(parts)):
            ptype, pmask = parts[i]
            
            if ptype == 0:  # clique
                if check_clique(pmask, v):
                    new_parts = parts[:]
                    new_parts[i] = (0, pmask | v_bit)
                    solve(assigned_mask | v_bit, new_parts, depth + 1)
            elif ptype == 1:  # independent set
                if check_indep(pmask, v):
                    new_parts = parts[:]
                    new_parts[i] = (1, pmask | v_bit)
                    solve(assigned_mask | v_bit, new_parts, depth + 1)
            else:  # undetermined (single vertex)
                # Try as clique
                if check_clique(pmask, v):
                    new_parts = parts[:]
                    new_parts[i] = (0, pmask | v_bit)
                    solve(assigned_mask | v_bit, new_parts, depth + 1)
                # Try as independent set
                if check_indep(pmask, v):
                    new_parts = parts[:]
                    new_parts[i] = (1, pmask | v_bit)
                    solve(assigned_mask | v_bit, new_parts, depth + 1)
        
        # Try starting a new part with v
        if len(parts) + 1 < best_k:
            new_parts = parts[:] + [(-1, v_bit)]
            solve(assigned_mask | v_bit, new_parts, depth + 1)
    
    # Initial state: no vertices assigned, no parts
    solve(0, [], 0)
    
    # If no solution found (shouldn't happen), fall back to greedy
    if best_parts is None:
        # Fallback: each vertex its own part
        best_parts = [[v] for v in range(n)]
    
    # Ensure types are correct: each part is list of ints
    result = []
    for part in best_parts:
        result.append([int(v) for v in part])
    
    return result
