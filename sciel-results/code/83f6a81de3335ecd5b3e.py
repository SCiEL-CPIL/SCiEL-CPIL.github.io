"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_polarity_graph_pg2_5():
    """
    Build the polarity graph of PG(2,5).
    Points: 31 points in PG(2,5).
    A point is adjacent to all points on its polar line.
    This graph is C4-free and has independence number 8.
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                p = [x, y, z]
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], 3, 5)
                        p = [(c * inv) % 5 for c in p]
                        break
                if p not in points:
                    points.append(p)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            p, q = points[i], points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    return edges, adj, n

def _to_bitmask(adj, n):
    """Convert adjacency list of sets to list of bitmasks."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    return masks

def _max_independent_set_size_bitmask(adj_mask, n, time_limit=1.0):
    """
    Find max independent set size using bitmask branch and bound with greedy coloring bound.
    adj_mask[i] has bit j set if i and j are adjacent.
    Independent set in G is Clique in complement.
    """
    start_time = time.time()
    best = [0]
    
    all_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)
    
    # Greedy coloring bound
    def _color_bound(candidates):
        """Return an upper bound on clique size using greedy coloring."""
        if not candidates:
            return 0
        
        # Extract vertices from bitmask
        cands = []
        temp = candidates
        while temp:
            lsb = temp & -temp
            cands.append(lsb.bit_length() - 1)
            temp ^= lsb
        
        if len(cands) <= best[0] + 1:
            return len(cands)
        
        # Order by degree in induced subgraph (descending)
        cands.sort(key=lambda v: -(comp_mask[v] & candidates).bit_count())
        
        # Greedy coloring
        colors = [-1] * n
        color_count = 0
        for v in cands:
            # Find smallest available color
            used_colors = set()
            vm = comp_mask[v] & candidates
            temp2 = vm
            while temp2:
                lsb = temp2 & -temp2
                w = lsb.bit_length() - 1
                if colors[w] >= 0:
                    used_colors.add(colors[w])
                temp2 ^= lsb
            
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
            if color_count >= len(cands):
                break
        
        return color_count

    def _search(candidates, current_size):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Quick bound
        rem = candidates.bit_count()
        if current_size + rem <= best[0]:
            return
        
        # Bound using greedy coloring
        color_bound = _color_bound(candidates)
        if current_size + color_bound <= best[0]:
            return
        
        # Pick vertex with max degree in candidate subgraph
        max_deg = -1
        v = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            idx = lsb.bit_length() - 1
            deg = (comp_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_cands ^= lsb
            
        if v == -1:
            return

        # Branch: include v first
        new_cands = candidates & comp_mask[v]
        _search(new_cands, current_size + 1)
        
        # Exclude v
        _search(candidates & ~(1 << v), current_size)

    _search(all_mask, 0)
    return best[0]

def _check_c4_free(adj, n):
    """Check if graph is C4-free."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) > 1:
                return False
    return True

def _get_maximal_compatibility_cliques(adj, n, time_limit=5.0):
    """
    Find maximal cliques in the compatibility graph H.
    H has vertices 0..n-1.
    u ~_H v if N_G(u) and N_G(v) are disjoint.
    """
    start_time = time.time()
    # Build compatibility graph
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & adj[j]):
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    cliques = []
    
    def bronk(R, P, X):
        if time.time() - start_time > time_limit:
            return
        if not P and not X:
            cliques.append(set(R))
            return
        
        # Choose pivot u from P union X
        u = -1
        max_deg = -1
        P_list = list(P)
        for v in P_list:
            deg = len(P & comp_adj[v])
            if deg > max_deg:
                max_deg = deg
                u = v
        
        if u == -1:
            if not X:
                cliques.append(set(R))
            return

        P_copy = set(P)
        for v in list(P_copy):
            if v in comp_adj[u]:
                continue
            
            new_R = R | {v}
            new_P = P & comp_adj[v]
            new_X = X & comp_adj[v]
            
            bronk(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)

    bronk(set(), set(range(n)), set())
    return cliques

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Build Base Graph
    edges, adj, n = _build_polarity_graph_pg2_5()
    
    if not _check_c4_free(adj, n):
        return {'n': n, 'edges': edges}
    
    best_n = n
    best_edges = edges
    best_adj = adj

    # 2. Extension Loop
    # We maintain adj as list of sets.
    # We need to check alpha(G \ N(v)) <= 9 for new vertex v connected to S.
    
    while time.time() < deadline - 20:
        # Generate candidate neighborhoods S (cliques in compatibility graph)
        # To save time, we limit the search for cliques or use a random subset
        # For n=31, finding all maximal cliques might be slow.
        # Let's try a heuristic: Randomly sample potential S sets.
        
        found_extension = False
        remaining_time = deadline - time.time()
        
        # Strategy: Try to find a valid S.
        # S must be a clique in compatibility graph.
        # We can generate random cliques or use Bron-Kerbosch with a time limit.
        
        # Optimization: Precompute incompatible pairs
        incompatible = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & adj[j]:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
        
        # Try random cliques
        max_trials = 500
        for _ in range(max_trials):
            if time.time() > deadline - 20:
                break
            
            # Pick a random start vertex
            start_v = random.randint(0, n - 1)
            S = {start_v}
            
            # Grow clique randomly
            available = [v for v in range(n) if v != start_v and v not in incompatible[start_v]]
            random.shuffle(available)
            
            for v in available:
                # Check if v is compatible with all in S
                valid = True
                for u in S:
                    if v in incompatible[u]:
                        valid = False
                        break
                if valid:
                    # Probabilistic add to keep S size manageable? 
                    # Or always add? 
                    # Let's always add to get maximal cliques? No, that's expensive.
                    # Let's add with probability 0.8
                    if random.random() < 0.8:
                        S.add(v)
            
            if len(S) < 1:
                continue
                
            # Check C4-freeness of S (already guaranteed by construction)
            # Check alpha constraint
            U = set(range(n)) - S
            if len(U) <= 9:
                alpha_u = 0
            else:
                u_list = sorted(U)
                u_map = {u: i for i, u in enumerate(u_list)}
                n_u = len(u_list)
                sub_adj_mask = [0] * n_u
                for i in range(n_u):
                    u = u_list[i]
                    mask = 0
                    for v in adj[u]:
                        if v in u_map:
                            mask |= (1 << u_map[v])
                    sub_adj_mask[i] = mask
                
                # Time limit for alpha check
                alpha_time = 0.5
                if n_u > 25:
                    alpha_time = 1.0
                elif n_u > 15:
                    alpha_time = 0.8
                
                alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=alpha_time)
            
            if alpha_u <= 9:
                # Valid extension
                new_v = n
                new_n = n + 1
                new_adj = [set(s) for s in adj]
                new_adj.append(set(S))
                for u in S:
                    new_adj[u].add(new_v)
                
                new_edges = list(edges)
                for u in S:
                    new_edges.append([u, new_v])
                
                edges = new_edges
                adj = new_adj
                n = new_n
                best_n = n
                best_edges = edges
                best_adj = adj
                
                found_extension = True
                break
        
        if not found_extension:
            # If random failed, try systematic search for smaller cliques?
            # Or just break?
            # Let's try a few more systematic ones
            # But for now, if random fails, we might be stuck.
            # We can try to restart with a different random seed or break.
            # Let's break to avoid wasting time if stuck.
            # Actually, let's try Bron-Kerbosch for a short time
            if time.time() < deadline - 30:
                cliques = _get_maximal_compatibility_cliques(adj, n, time_limit=5.0)
                if cliques:
                    # Sort by size descending
                    cliques.sort(key=lambda s: -len(s))
                    for S in cliques[:20]:
                        if time.time() > deadline - 20:
                            break
                        U = set(range(n)) - S
                        if len(U) <= 9:
                            alpha_u = 0
                        else:
                            u_list = sorted(U)
                            u_map = {u: i for i, u in enumerate(u_list)}
                            n_u = len(u_list)
                            sub_adj_mask = [0] * n_u
                            for i in range(n_u):
                                u = u_list[i]
                                mask = 0
                                for v in adj[u]:
                                    if v in u_map:
                                        mask |= (1 << u_map[v])
                                sub_adj_mask[i] = mask
                            alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=1.0)
                        
                        if alpha_u <= 9:
                            new_v = n
                            new_n = n + 1
                            new_adj = [set(s) for s in adj]
                            new_adj.append(set(S))
                            for u in S:
                                new_adj[u].add(new_v)
                            
                            new_edges = list(edges)
                            for u in S:
                                new_edges.append([u, new_v])
                            
                            edges = new_edges
                            adj = new_adj
                            n = new_n
                            best_n = n
                            best_edges = edges
                            best_adj = new_adj
                            
                            found_extension = True
                            break
            
            if not found_extension:
                # Give up on extending further
                break

    # Final Validation
    if time.time() < deadline - 5 and best_n > 1:
        if not _check_c4_free(best_adj, best_n):
            return {'n': n, 'edges': edges}
        
        adj_mask_final = _to_bitmask(best_adj, best_n)
        alpha_final = _max_independent_set_size_bitmask(adj_mask_final, best_n, time_limit=3.0)
        
        if alpha_final >= 11:
            # Invalid, return previous best? 
            # The loop only adds if valid, so this shouldn't happen unless bug.
            # But if it does, we might have a smaller valid one?
            # For safety, return the last verified one? 
            # We don't store history. 
            # Let's just return the current one, hoping the check was wrong? 
            # No, if alpha >= 11, it's invalid. 
            # We should have a fallback. 
            # Let's return the base graph if this fails.
            return {'n': 31, 'edges': _build_polarity_graph_pg2_5()[0]}

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
