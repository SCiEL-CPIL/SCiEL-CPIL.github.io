"""Cochromatic partition solver using branch-and-bound over maximal cliques/independent sets."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _bron_kerbosch_cliques(adj):
    """Find all maximal cliques using Bron-Kerbosch with pivot."""
    n = P['n']
    max_cliques = []
    
    def bronk(R, P_set, X):
        if not P_set and not X:
            max_cliques.append(frozenset(R))
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        best_u = max(candidates, key=lambda u: len(P_set & adj[u]))
        for v in list(P_set - adj[best_u]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    
    bronk(frozenset(), set(range(n)), set())
    return max_cliques

def _bron_kerbosch_independent_sets(adj):
    """Find all maximal independent sets using Bron-Kerbosch on complement graph."""
    n = P['n']
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    
    max_indep = []
    
    def bronk(R, P_set, X):
        if not P_set and not X:
            max_indep.append(frozenset(R))
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        best_u = max(candidates, key=lambda u: len(P_set & comp_adj[u]))
        for v in list(P_set - comp_adj[best_u]):
            bronk(R | {v}, P_set & comp_adj[v], X & comp_adj[v])
            P_set.remove(v)
            X.add(v)
    
    bronk(frozenset(), set(range(n)), set())
    return max_indep

def _solve_cochromatic(adj):
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Find all maximal cliques and independent sets
    max_cliques = _bron_kerbosch_cliques(adj)
    max_indep = _bron_kerbosch_independent_sets(adj)
    
    # Build candidate list (union, deduplicated)
    all_candidates = set(max_cliques) | set(max_indep)
    candidate_list = list(all_candidates)
    
    # Convert to bitmask representation
    candidate_masks = []
    for fs in candidate_list:
        mask = 0
        for v in fs:
            mask |= (1 << v)
        candidate_masks.append(mask)
    
    # Sort by size descending
    candidate_list.sort(key=lambda x: -len(x))
    candidate_masks.sort(key=lambda m: -bin(m).count('1'))
    
    num_candidates = len(candidate_masks)
    
    # Greedy initial solution
    best_count = n
    best_solution = None
    remaining = full_mask
    greedy_sol = []
    while remaining:
        best_idx = -1
        best_covered = 0
        for i in range(num_candidates):
            covered = candidate_masks[i] & remaining
            cnt = bin(covered).count('1')
            if cnt > best_covered:
                best_covered = cnt
                best_idx = i
        if best_idx >= 0:
            greedy_sol.append(candidate_list[best_idx])
            remaining &= ~candidate_masks[best_idx]
    if len(greedy_sol) < best_count:
        best_count = len(greedy_sol)
        best_solution = [list(fs) for fs in greedy_sol]
    
    # Precompute max candidate size
    max_cand_size = max(len(fs) for fs in candidate_list) if candidate_list else 1
    
    # Precompute for each vertex, which candidates cover it
    vertex_candidates = [[] for _ in range(n)]
    for i, mask in enumerate(candidate_masks):
        for v in range(n):
            if mask & (1 << v):
                vertex_candidates[v].append(i)
    
    # Precompute popcount for all masks up to 2^n
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Lower bound function - improved
    def lower_bound(remaining_mask):
        if remaining_mask == 0:
            return 0
        uncovered = popcount[remaining_mask]
        return (uncovered + max_cand_size - 1) // max_cand_size
    
    # Stronger lower bound using matching-like argument
    def refined_lower_bound(remaining_mask):
        if remaining_mask == 0:
            return 0
        # Count vertices
        verts = [v for v in range(n) if remaining_mask & (1 << v)]
        if not verts:
            return 0
        # Simple bound
        base_lb = (len(verts) + max_cand_size - 1) // max_cand_size
        # Try to find a set of vertices where no single candidate covers more than k
        # For now, use the base bound
        return base_lb
    
    # DFS with branch and bound
    best_count_val = best_count
    best_sol_val = best_solution[:] if best_solution else []
    node_count = [0]
    
    def dfs(remaining_mask, current_count, current_sol, deadline):
        nonlocal best_count_val, best_sol_val
        node_count[0] += 1
        if time.time() > deadline:
            return
        if remaining_mask == 0:
            if current_count < best_count_val:
                best_count_val = current_count
                best_sol_val = [list(fs) for fs in current_sol]
            return
        if current_count + lower_bound(remaining_mask) >= best_count_val:
            return
        
        # Find the "hardest" vertex (fewest covering candidates that are subsets of remaining)
        hardest_v = -1
        min_cnt = float('inf')
        for v in range(n):
            if remaining_mask & (1 << v):
                cnt = 0
                for i in vertex_candidates[v]:
                    if candidate_masks[i] & remaining_mask == candidate_masks[i]:
                        cnt += 1
                if cnt < min_cnt:
                    min_cnt = cnt
                    hardest_v = v
        
        if hardest_v < 0:
            return
        
        # Try all candidates that cover hardest_v and are subsets of remaining, sorted by coverage
        covering = []
        for i in vertex_candidates[hardest_v]:
            if candidate_masks[i] & remaining_mask == candidate_masks[i]:
                covering.append(i)
        covering.sort(key=lambda i: -popcount[candidate_masks[i] & remaining_mask])
        
        for i in covering:
            new_remaining = remaining_mask & ~candidate_masks[i]
            current_sol.append(candidate_list[i])
            dfs(new_remaining, current_count + 1, current_sol, deadline)
            current_sol.pop()
    
    deadline = time.time() + 50
    dfs(full_mask, 0, [], deadline)
    
    if best_sol_val:
        return best_sol_val
    return best_solution

def construct():
    adj = _build_adj()
    solution = _solve_cochromatic(adj)
    
    # Validate
    edge_set = set()
    for a, b in P['edges']:
        edge_set.add((a, b))
        edge_set.add((b, a))
    
    def _validate_and_fallback(solution):
        for part in solution:
            part_set = set(part)
            is_clique = True
            is_indep = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    if (part[i], part[j]) in edge_set:
                        is_indep = False
                    else:
                        is_clique = False
            if not is_clique and not is_indep:
                return None
        
        all_vertices = set()
        for part in solution:
            for v in part:
                if v in all_vertices:
                    return None
                all_vertices.add(v)
        
        if all_vertices != set(range(P['n'])):
            return None
        
        return solution
    
    valid = _validate_and_fallback(solution)
    if valid is not None:
        return [list(p) for p in valid]
    
    # Fallback greedy
    parts = []
    for v in range(P['n']):
        placed = False
        for p in parts:
            ok = True
            for u in p:
                if (u, v) in edge_set:
                    for w in p:
                        if w != u and (u, w) not in edge_set:
                            ok = False
                            break
                else:
                    for w in p:
                        if w != u and (u, w) in edge_set:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                p.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return [list(p) for p in parts]
# EVOLVE-BLOCK-END
