"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Precompute edge map
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[idx] = (u, v)
            idx += 1
            
    # Reverse map: (u, v) -> index
    edge_index = {}
    for i, (u, v) in edge_map.items():
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Structural Initialization: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    
    current_edges = set()
    for u in A:
        for v in B:
            current_edges.add(edge_index[(u, v)])
            
    def build_state(edge_set):
        """Build adjacency list and degree array from edge set."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def check_edge_validity(u, v, adj, deg):
        """Check if edge (u,v) causes a violation given current adj/deg.
        A violation exists if there are s in N(u)\\{v} and t in N(v)\\{u}
        with s != t and deg[s] == deg[t]."""
        s_candidates = [s for s in adj[u] if s != v]
        t_candidates = [t for t in adj[v] if t != u]
        
        if not s_candidates or not t_candidates:
            return True
        
        # Build degree -> set mapping for s candidates
        s_by_deg = {}
        for s in s_candidates:
            d = deg[s]
            if d not in s_by_deg:
                s_by_deg[d] = set()
            s_by_deg[d].add(s)
        
        # Check each t candidate
        for t in t_candidates:
            d = deg[t]
            if d in s_by_deg:
                S_set = s_by_deg[d]
                # Violation if there exists s in S_set with s != t
                if len(S_set) > 1 or (len(S_set) == 1 and list(S_set)[0] != t):
                    return False
        
        return True

    def is_valid(edge_set):
        """Full validity check for an edge set."""
        adj, deg = build_state(edge_set)
        for e in edge_set:
            u, v = edge_map[e]
            if not check_edge_validity(u, v, adj, deg):
                return False
        return True

    # Verify initial state
    if not is_valid(current_edges):
        star_edges = set()
        for v in range(1, n):
            star_edges.add(edge_index[(0, v)])
        return [[u, v] for e in star_edges for u, v in [edge_map[e]]]

    best_edges = current_edges.copy()
    best_score = len(current_edges)
    
    # Maintain state for incremental updates
    adj, deg = build_state(current_edges)
    
    def try_add_edge(idx, edge_set, adj, deg):
        """Try adding edge idx to edge_set. Returns (success, new_adj, new_deg)."""
        u, v = edge_map[idx]
        # Temporarily update degrees
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
        # Check if this addition causes any violations
        # Only need to check edges that are affected:
        # 1. The new edge (u,v) itself
        # 2. Edges incident to u (their s-candidates changed)
        # 3. Edges incident to v (their t-candidates changed)
        
        # Check the new edge (u,v)
        if not check_edge_validity(u, v, adj, deg):
            # Roll back
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1
            return False, adj, deg
        
        # Check all other edges incident to u
        for s in list(adj[u]):
            if s == v:
                continue
            # Edge (s, u): s is now a candidate for some paths
            # We need to check if edge (s,u) is still valid
            # For edge (s,u): candidates for "other end" are N(s)\{u} and N(u)\{s}
            # N(u)\{s} now includes v (new neighbor)
            if not check_edge_validity(s, u, adj, deg):
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
                return False, adj, deg
        
        # Check all other edges incident to v
        for t in list(adj[v]):
            if t == u:
                continue
            if not check_edge_validity(v, t, adj, deg):
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
                return False, adj, deg
        
        return True, adj, deg

    def try_remove_edge(idx, edge_set, adj, deg):
        """Remove edge idx from state. Returns (new_adj, new_deg)."""
        u, v = edge_map[idx]
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
        return adj, deg

    # Greedy Add with incremental validation
    # Order edges by "conflict potential": prefer edges between vertices with different degree profiles
    missing_internal = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            u, v = A[i], A[j]
            missing_internal.append(edge_index[(u, v)])
            
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            u, v = B[i], B[j]
            missing_internal.append(edge_index[(u, v)])
            
    random.shuffle(missing_internal)
    
    for idx in missing_internal:
        if time.time() > deadline - 3:
            break
        success, adj, deg = try_add_edge(idx, current_edges, adj, deg)
        if success:
            current_edges.add(idx)
            if len(current_edges) > best_score:
                best_edges = current_edges.copy()
                best_score = len(current_edges)
    
    # Second pass: try all remaining missing edges
    all_missing = [i for i in range(total_edges) if i not in current_edges]
    random.shuffle(all_missing)
    
    for idx in all_missing:
        if time.time() > deadline - 3:
            break
        success, adj, deg = try_add_edge(idx, current_edges, adj, deg)
        if success:
            current_edges.add(idx)
            if len(current_edges) > best_score:
                best_edges = current_edges.copy()
                best_score = len(current_edges)

    # Local search with 1-swap and 2-opt moves
    improved = True
    swap_count = 0
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Rebuild state to ensure consistency
        adj, deg = build_state(current_edges)
        
        # Try 1-swap moves: remove edge e1, add edge e2
        existing_edges = list(current_edges)
        missing_edges = [i for i in range(total_edges) if i not in current_edges]
        
        random.shuffle(existing_edges)
        random.shuffle(missing_edges)
        
        # Limit the number of swaps we try per iteration to manage time
        max_swaps_to_try = min(len(existing_edges) * len(missing_edges), 5000)
        attempts = 0
        
        for e1 in existing_edges:
            if time.time() > deadline - 5 or attempts >= max_swaps_to_try:
                break
            for e2 in missing_edges:
                if time.time() > deadline - 5 or attempts >= max_swaps_to_try:
                    break
                attempts += 1
                
                # Try: remove e1, add e2
                u1, v1 = edge_map[e1]
                u2, v2 = edge_map[e2]
                
                # Remove e1
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                deg[u1] -= 1
                deg[v1] -= 1
                
                # Add e2
                adj[u2].add(v2)
                adj[v2].add(u2)
                deg[u2] += 1
                deg[v2] += 1
                
                # Check validity of affected edges
                valid = True
                affected = set()
                affected.add(e2)  # new edge
                for s in adj[u2]:
                    if s != v2:
                        affected.add(edge_index[(s, u2)])
                for t in adj[v2]:
                    if t != u2:
                        affected.add(edge_index[(v2, t)])
                for s in adj[u1]:
                    if s != v1:
                        affected.add(edge_index[(s, u1)])
                for t in adj[v1]:
                    if t != u1:
                        affected.add(edge_index[(v1, t)])
                
                for e in affected:
                    if e not in current_edges and e != e2:
                        continue  # edge not present, skip
                    u, v = edge_map[e]
                    if not check_edge_validity(u, v, adj, deg):
                        valid = False
                        break
                
                if valid:
                    # Also need to check that removing e1 didn't break other edges
                    # But removing an edge only reduces constraints, so it's safe
                    current_edges.discard(e1)
                    current_edges.add(e2)
                    if len(current_edges) > best_score:
                        best_edges = current_edges.copy()
                        best_score = len(current_edges)
                    improved = True
                    # Rebuild state for next iteration
                    adj, deg = build_state(current_edges)
                    missing_edges = [i for i in range(total_edges) if i not in current_edges]
                    existing_edges = list(current_edges)
                    break
                else:
                    # Roll back
                    adj[u2].discard(v2)
                    adj[v2].discard(u2)
                    deg[u2] -= 1
                    deg[v2] -= 1
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    deg[u1] += 1
                    deg[v1] += 1
        
        # If no improvement from 1-swap, try 2-opt: remove 2, add 2
        if not improved and time.time() < deadline - 8:
            existing_edges = list(current_edges)
            missing_edges = [i for i in range(total_edges) if i not in current_edges]
            
            if len(existing_edges) >= 2 and len(missing_edges) >= 2:
                random.shuffle(existing_edges)
                random.shuffle(missing_edges)
                
                # Try a limited number of 2-opt moves
                max_2opt = min(len(existing_edges) * len(missing_edges), 2000)
                attempts = 0
                
                for i in range(len(existing_edges)):
                    if time.time() > deadline - 8 or attempts >= max_2opt:
                        break
                    for j in range(i+1, len(existing_edges)):
                        if time.time() > deadline - 8 or attempts >= max_2opt:
                            break
                        e1 = existing_edges[i]
                        e2 = existing_edges[j]
                        
                        for k in range(len(missing_edges)):
                            if time.time() > deadline - 8 or attempts >= max_2opt:
                                break
                            for l in range(k+1, len(missing_edges)):
                                if time.time() > deadline - 8 or attempts >= max_2opt:
                                    break
                                attempts += 1
                                a = missing_edges[k]
                                b = missing_edges[l]
                                
                                # Remove e1, e2 and add a, b
                                u1, v1 = edge_map[e1]
                                u2, v2 = edge_map[e2]
                                u3, v3 = edge_map[a]
                                u4, v4 = edge_map[b]
                                
                                adj[u1].discard(v1)
                                adj[v1].discard(u1)
                                deg[u1] -= 1
                                deg[v1] -= 1
                                adj[u2].discard(v2)
                                adj[v2].discard(u2)
                                deg[u2] -= 1
                                deg[v2] -= 1
                                
                                adj[u3].add(v3)
                                adj[v3].add(u3)
                                deg[u3] += 1
                                deg[v3] += 1
                                adj[u4].add(v4)
                                adj[v4].add(u4)
                                deg[u4] += 1
                                deg[v4] += 1
                                
                                # Check all edges for validity
                                valid = True
                                for e in current_edges:
                                    if e == e1 or e == e2:
                                        continue
                                    u, v = edge_map[e]
                                    if not check_edge_validity(u, v, adj, deg):
                                        valid = False
                                        break
                                # Also check new edges
                                if valid:
                                    if not check_edge_validity(u3, v3, adj, deg):
                                        valid = False
                                if valid:
                                    if not check_edge_validity(u4, v4, adj, deg):
                                        valid = False
                                
                                if valid:
                                    current_edges.discard(e1)
                                    current_edges.discard(e2)
                                    current_edges.add(a)
                                    current_edges.add(b)
                                    if len(current_edges) > best_score:
                                        best_edges = current_edges.copy()
                                        best_score = len(current_edges)
                                    improved = True
                                    adj, deg = build_state(current_edges)
                                    break
                                else:
                                    # Roll back
                                    adj[u3].discard(v3)
                                    adj[v3].discard(u4)
                                    deg[u3] -= 1
                                    deg[v3] -= 1
                                    adj[u4].discard(v4)
                                    adj[v4].discard(u4)
                                    deg[u4] -= 1
                                    deg[v4] -= 1
                                    adj[u1].add(v1)
                                    adj[v1].add(u1)
                                    deg[u1] += 1
                                    deg[v1] += 1
                                    adj[u2].add(v2)
                                    adj[v2].add(u2)
                                    deg[u2] += 1
                                    deg[v2] += 1
                                
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
        
        # If still no improvement, try random perturbation
        if not improved and time.time() < deadline - 5:
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                if not current_edges:
                    break
                # Remove 1-3 random edges
                num_remove = random.randint(1, 3)
                to_remove = random.sample(list(current_edges), min(num_remove, len(current_edges)))
                perturbed = current_edges.copy()
                for e in to_remove:
                    perturbed.discard(e)
                
                # Greedy add back
                p_adj, p_deg = build_state(perturbed)
                missing = [i for i in range(total_edges) if i not in perturbed]
                random.shuffle(missing)
                
                for idx in missing:
                    if time.time() > deadline - 5:
                        break
                    success, p_adj, p_deg = try_add_edge(idx, perturbed, p_adj, p_deg)
                    if success:
                        perturbed.add(idx)
                
                if len(perturbed) > best_score:
                    current_edges = perturbed
                    best_edges = current_edges.copy()
                    best_score = len(current_edges)
                    improved = True
                    adj, deg = build_state(current_edges)

    # Final validation
    if not is_valid(best_edges):
         star_edges = set()
         for v in range(1, n):
            star_edges.add(edge_index[(0, v)])
         best_edges = star_edges

    result = []
    for e in best_edges:
        u, v = edge_map[e]
        result.append([u, v])
        
    return result
# EVOLVE-BLOCK-END
