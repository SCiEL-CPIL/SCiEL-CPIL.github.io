"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_polarity_graph_pg2_q(q):
    """
    Build the polarity graph of PG(2,q) using a symmetric bilinear form.
    This graph is C4-free by algebraic properties of the polarity.
    n = q^2 + q + 1 vertices.
    """
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                p = [x, y, z]
                # Normalize: first non-zero coordinate to 1
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], q-2, q)
                        p = [(c * inv) % q for c in p]
                        break
                if p not in points:
                    points.append(p)
    
    n = len(points)
    assert n == q*q + q + 1, f"Expected {q*q+q+1} points, got {n}"
    
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            p, r = points[i], points[j]
            # Use the standard dot product form for simplicity and C4-freedom
            # Actually, for PG(2,q), the polarity graph is defined by the orthogonal relation.
            # A common C4-free graph is the one where u~v if <u,v> = 0.
            # For q=7, n=57.
            dot = (p[0]*r[2] + p[2]*r[0] + p[1]*r[1]) % q
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    return edges, adj, n, points

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

    def _color_bound(candidates):
        """Return an upper bound on clique size in the subgraph induced by candidates."""
        if not candidates:
            return 0
        
        cands_list = []
        temp = candidates
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cands_list.append(idx)
            temp ^= lsb
        
        cands_list.sort(key=lambda v: -(comp_mask[v] & candidates).bit_count())
        
        color_of = [-1] * len(cands_list)
        color_count = 0
        
        for idx, v in enumerate(cands_list):
            used_colors = set()
            v_comp = comp_mask[v]
            for jdx in range(idx):
                w = cands_list[jdx]
                if (v_comp >> w) & 1:
                    used_colors.add(color_of[jdx])
            c = 0
            while c in used_colors:
                c += 1
            color_of[idx] = c
            if c + 1 > color_count:
                color_count = c + 1
        return color_count

    def _search(candidates, current_size):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        color_bound = _color_bound(candidates)
        if current_size + color_bound <= best[0]:
            return
        
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

        new_cands = candidates & comp_mask[v]
        _search(new_cands, current_size + 1)
        
        new_cands2 = candidates & ~(1 << v)
        _search(new_cands2, current_size)

    _search((1 << n) - 1, 0)
    return best[0]

def _check_c4_free(adj, n):
    """Check if graph is C4-free."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) > 1:
                return False
    return True

def _induced_subgraph(adj, n, vertex_indices):
    """Extract induced subgraph on given vertex indices, relabeled to 0..k-1."""
    k = len(vertex_indices)
    old_to_new = {old: new for new, old in enumerate(vertex_indices)}
    new_adj = [set() for _ in range(k)]
    new_edges = []
    for i in range(k):
        for j in range(i+1, k):
            if vertex_indices[j] in adj[vertex_indices[i]]:
                new_adj[i].add(j)
                new_adj[j].add(i)
                new_edges.append([i, j])
    return new_edges, new_adj, k

def _greedy_independent_set(adj, n):
    """Greedy: repeatedly pick vertex with minimum degree in remaining."""
    remaining = set(range(n))
    size = 0
    while remaining:
        min_deg = float('inf')
        chosen = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                chosen = v
        size += 1
        remaining.discard(chosen)
        remaining -= adj[chosen]
    return size

def _try_extend_with_vertex(adj, n, edges, time_limit=10.0):
    """
    Try to add one vertex to the graph by finding a valid neighbor set.
    Returns (new_edges, new_adj, new_n) or None.
    """
    start_time = time.time()
    
    # Build compatibility graph: u~v if N(u) ∩ N(v) = ∅
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & adj[j]):
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    # Find maximal cliques in compatibility graph (candidate neighbor sets)
    cliques = []
    
    def bronk_kerbosch(R, P, X):
        if time.time() - start_time > time_limit:
            return
        if not P and not X:
            cliques.append(frozenset(R))
            return
        
        if P or X:
            P_X = P | X
            pivot = max(P_X, key=lambda u: len(P & comp_adj[u]))
            candidates = [v for v in P if v not in comp_adj[pivot]]
        else:
            candidates = list(P)
        
        for v in candidates:
            bronk_kerbosch(R | {v}, P & comp_adj[v], X & comp_adj[v])
            P.discard(v)
            X.add(v)
    
    bronk_kerbosch(set(), set(range(n)), set())
    
    # Sort by size descending
    cliques.sort(key=lambda s: -len(s))
    
    for S in cliques:
        if time.time() - start_time > time_limit:
            break
        
        U = set(range(n)) - S
        u_list = sorted(U)
        n_u = len(u_list)
        
        if n_u == 0:
            alpha_u = 0
        else:
            # Quick greedy check first
            remaining = set(u_list)
            greedy_size = 0
            while remaining:
                min_deg = float('inf')
                chosen = None
                for v in remaining:
                    deg = len(adj[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                if chosen is None:
                    break
                greedy_size += 1
                remaining.discard(chosen)
                remaining -= adj[chosen]
            
            if greedy_size > 9:
                continue
            
            sub_adj_mask = [0] * n_u
            for i in range(n_u):
                for j in range(i+1, n_u):
                    u, v = u_list[i], u_list[j]
                    if v in adj[u]:
                        sub_adj_mask[i] |= (1 << j)
                        sub_adj_mask[j] |= (1 << i)
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
            return new_edges, new_adj, new_n
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Strategy: Build PG(2,7) polarity graph (57 vertices, C4-free, small alpha)
    # Then take a large induced subgraph (up to 43 vertices)
    # Since any induced subgraph of a C4-free graph is C4-free,
    # and alpha of subgraph <= alpha of full graph, this is valid.
    
    best_result = None
    
    # Try PG(2,7) first - 57 vertices
    try:
        edges7, adj7, n7, points7 = _build_polarity_graph_pg2_q(7)
        
        if _check_c4_free(adj7, n7):
            # Check alpha of full graph
            adj_mask7 = _to_bitmask(adj7, n7)
            alpha7 = _max_independent_set_size_bitmask(adj_mask7, n7, time_limit=3.0)
            
            if alpha7 <= 10:
                # Take a 43-vertex induced subgraph
                # Use vertices with highest degree first for better structure
                degrees7 = [len(adj7[i]) for i in range(n7)]
                sorted_verts = sorted(range(n7), key=lambda v: -degrees7[v])
                
                for target_n in [43, 42, 41, 40, 39]:
                    if time.time() > deadline - 10:
                        break
                    sel = sorted_verts[:target_n]
                    sub_edges, sub_adj, sub_n = _induced_subgraph(adj7, n7, sel)
                    
                    # Verify C4-free (should be, but check)
                    if not _check_c4_free(sub_adj, sub_n):
                        continue
                    
                    # Verify alpha
                    sub_mask = _to_bitmask(sub_adj, sub_n)
                    sub_alpha = _max_independent_set_size_bitmask(sub_mask, sub_n, time_limit=2.0)
                    
                    if sub_alpha <= 10:
                        best_result = {'n': sub_n, 'edges': sub_edges}
                        break
                else:
                    # Fallback: just take 39
                    sel = sorted_verts[:39]
                    sub_edges, sub_adj, sub_n = _induced_subgraph(adj7, n7, sel)
                    if _check_c4_free(sub_adj, sub_n):
                        sub_mask = _to_bitmask(sub_adj, sub_n)
                        sub_alpha = _max_independent_set_size_bitmask(sub_mask, sub_n, time_limit=2.0)
                        if sub_alpha <= 10:
                            best_result = {'n': sub_n, 'edges': sub_edges}
        
        # If we have a good result from PG(2,7), try to extend further
        if best_result and time.time() < deadline - 30:
            # Try to add more vertices to the best subgraph
            cur_n = best_result['n']
            cur_edges = best_result['edges']
            cur_adj = [set() for _ in range(cur_n)]
            for e in cur_edges:
                cur_adj[e[0]].add(e[1])
                cur_adj[e[1]].add(e[0])
            
            while time.time() < deadline - 15:
                ext = _try_extend_with_vertex(cur_adj, cur_n, cur_edges, time_limit=15.0)
                if ext is None:
                    break
                cur_edges, cur_adj, cur_n = ext
                best_result = {'n': cur_n, 'edges': cur_edges}
                if cur_n >= 43:
                    break
        
        if best_result:
            return best_result
    
    except Exception:
        pass
    
    # Fallback: PG(2,5) polarity graph (31 vertices)
    try:
        edges5, adj5, n5, points5 = _build_polarity_graph_pg2_q(5)
        
        if _check_c4_free(adj5, n5):
            adj_mask5 = _to_bitmask(adj5, n5)
            alpha5 = _max_independent_set_size_bitmask(adj_mask5, n5, time_limit=2.0)
            
            if alpha5 <= 10:
                # Try to extend
                cur_n = n5
                cur_edges = edges5
                cur_adj = adj5
                
                while time.time() < deadline - 15:
                    ext = _try_extend_with_vertex(cur_adj, cur_n, cur_edges, time_limit=15.0)
                    if ext is None:
                        break
                    cur_edges, cur_adj, cur_n = ext
                    if cur_n >= 43:
                        break
                
                # Final validation
                if _check_c4_free(cur_adj, cur_n):
                    cur_mask = _to_bitmask(cur_adj, cur_n)
                    cur_alpha = _max_independent_set_size_bitmask(cur_mask, cur_n, time_limit=3.0)
                    if cur_alpha <= 10:
                        return {'n': cur_n, 'edges': cur_edges}
                
                # Return the base if extension failed validation
                if _check_c4_free(adj5, n5):
                    return {'n': n5, 'edges': edges5}
    except Exception:
        pass
    
    # Ultimate fallback: empty graph on 1 vertex
    return {'n': 1, 'edges': []}

# EVOLVE-BLOCK-END
