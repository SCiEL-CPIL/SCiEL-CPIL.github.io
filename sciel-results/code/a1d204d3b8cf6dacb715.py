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

    # Precompute vertex ordering by degree in complement graph for better pruning
    comp_deg = [comp_mask[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda i: -comp_deg[i])
    
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
        
        colors = {}
        color_count = 0
        for v in cands_list:
            used = set()
            for w in cands_list:
                if w != v and w in colors and (comp_mask[v] >> w) & 1:
                    used.add(colors[w])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
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

def _find_valid_neighbor_set(adj, n, time_limit=5.0):
    """
    Systematically search for a valid neighbor set S such that:
    1. S is a clique in the compatibility graph (no two vertices in S share a common neighbor in G)
    2. The independence number of G - S is at most 9
    """
    start_time = time.time()
    
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & adj[j]):
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    cliques = []
    
    def bronk_kerbosch(R, P, X):
        if time.time() - start_time > time_limit:
            return
        if not P and not X:
            if len(R) >= 1:
                cliques.append(set(R))
            return
        
        u = -1
        max_deg = -1
        for v in list(P) + list(X):
            deg = len(P & comp_adj[v])
            if deg > max_deg:
                max_deg = deg
                u = v
        
        if u == -1:
            return
            
        for v in list(P):
            if v in comp_adj[u]:
                continue
            
            new_R = R | {v}
            new_P = P & comp_adj[v]
            new_X = X & comp_adj[v]
            
            bronk_kerbosch(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)
    
    bronk_kerbosch(set(), set(range(n)), set())
    
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
            u_map = {u: i for i, u in enumerate(u_list)}
            sub_adj_mask = [0] * n_u
            for i in range(n_u):
                for j in range(i+1, n_u):
                    u, v = u_list[i], u_list[j]
                    if v in adj[u]:
                        sub_adj_mask[i] |= (1 << j)
                        sub_adj_mask[j] |= (1 << i)
            alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=1.0)
        
        if alpha_u <= 9:
            return S
    
    return None

def _generate_random_c4_free_seed(n_target=30, time_limit=5.0):
    """Generate a random C4-free graph with low alpha."""
    start_time = time.time()
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        adj = [set() for _ in range(n_target)]
        for i in range(n_target):
            for j in range(i+1, n_target):
                if random.random() < 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Remove C4s
        changed = True
        while changed:
            changed = False
            for i in range(n_target):
                for j in range(i+1, n_target):
                    common = adj[i] & adj[j]
                    if len(common) > 1:
                        # Remove one edge
                        k = list(common)[0]
                        adj[i].remove(k)
                        adj[k].remove(i)
                        adj[j].remove(k)
                        adj[k].remove(j)
                        changed = True
                        break
                if changed:
                    break
        
        if not _check_c4_free(adj, n_target):
            continue
            
        adj_mask = _to_bitmask(adj, n_target)
        alpha = _max_independent_set_size_bitmask(adj_mask, n_target, time_limit=1.0)
        if alpha <= 7:
            edges = []
            for i in range(n_target):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            return edges, adj, n_target
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_n = 0
    best_edges = []
    best_adj = []

    # Try PG(2,5) seed
    edges, adj, n = _build_polarity_graph_pg2_5()
    if not _check_c4_free(adj, n):
        return {'n': n, 'edges': edges}
    
    adj_mask = _to_bitmask(adj, n)
    alpha = _max_independent_set_size_bitmask(adj_mask, n, time_limit=2.0)
    
    if alpha < 11:
        best_n = n
        best_edges = edges
        best_adj = adj

        # Extension loop
        while time.time() < deadline - 15:
            S = _find_valid_neighbor_set(adj, n, time_limit=min(10.0, deadline - time.time() - 20))
            if S is not None:
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
                best_edges = new_edges
                best_adj = new_adj
            else:
                break

    # Try Random Seeds if we have time
    if time.time() < deadline - 30 and best_n < 39:
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            seed_res = _generate_random_c4_free_seed(30, time_limit=5.0)
            if seed_res:
                s_edges, s_adj, s_n = seed_res
                # Extend this seed
                cur_edges, cur_adj, cur_n = s_edges, s_adj, s_n
                while time.time() < deadline - 15:
                    S = _find_valid_neighbor_set(cur_adj, cur_n, time_limit=min(5.0, deadline - time.time() - 20))
                    if S is not None:
                        new_v = cur_n
                        new_n = cur_n + 1
                        new_adj = [set(s) for s in cur_adj]
                        new_adj.append(set(S))
                        for u in S:
                            new_adj[u].add(new_v)
                        
                        new_edges = list(cur_edges)
                        for u in S:
                            new_edges.append([u, new_v])
                        
                        cur_edges = new_edges
                        cur_adj = new_adj
                        cur_n = new_n
                        
                        if cur_n > best_n:
                            best_n = cur_n
                            best_edges = cur_edges
                            best_adj = cur_adj
                    else:
                        break

    # Final Validation
    if best_n > 1:
        if not _check_c4_free(best_adj, best_n):
            return {'n': n, 'edges': edges}
        
        adj_mask_final = _to_bitmask(best_adj, best_n)
        alpha_final = _max_independent_set_size_bitmask(adj_mask_final, best_n, time_limit=5.0)
        
        if alpha_final >= 11:
            return {'n': n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
