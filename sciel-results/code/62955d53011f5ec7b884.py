# EVOLVE-BLOCK-START
import time
import random

def _edges_to_adj(edge_set, n=18):
    adj = [0] * n
    for (u, v) in edge_set:
        if u == v:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free_adj(adj, n=18):
    """Fast K4-free check using bitmasks."""
    for u in range(n):
        au = adj[u]
        v = u + 1
        while v < n:
            if (au >> v) & 1:
                common = au & adj[v]
                cm = common & ~((1 << (v + 1)) - 1)
                while cm:
                    lsb = cm & (-cm)
                    w = lsb.bit_length() - 1
                    if adj[w] & (common & ~(1 << w)):
                        return False
                    cm ^= lsb
            v += 1
    return True

def _is_k4_free(edge_set, n=18):
    adj = _edges_to_adj(edge_set, n)
    return _is_k4_free_adj(adj, n)

def _max_triangle_free_subgraph_size_adj(adj, n=18, deadline=None):
    """
    Find the size of the largest triangle-free induced subgraph.
    Uses branch and bound on the minimum vertex cover of triangles.
    """
    full = (1 << n) - 1
    
    triangles = []
    for u in range(n):
        au = adj[u]
        v = u + 1
        while v < n:
            if (au >> v) & 1:
                common = au & adj[v] & ~((1 << (v + 1)) - 1)
                cm = common
                while cm:
                    lsb = cm & (-cm)
                    w = lsb.bit_length() - 1
                    triangles.append((1 << u) | (1 << v) | (1 << w))
                    cm ^= lsb
            v += 1
    
    if not triangles:
        return n
    
    # Greedy vertex cover
    remaining_tris = list(triangles)
    removed = 0
    while remaining_tris:
        counts = [0] * n
        for tri in remaining_tris:
            for v in range(n):
                if (tri >> v) & 1:
                    counts[v] += 1
        best_v = -1
        best_c = 0
        for v in range(n):
            if (removed >> v) & 1:
                continue
            if counts[v] > best_c:
                best_c = counts[v]
                best_v = v
        if best_v == -1 or best_c == 0:
            break
        removed |= (1 << best_v)
        remaining_tris = [t for t in remaining_tris if not (t & (1 << best_v))]
    
    greedy_cover = bin(removed).count("1")
    lower_bound_tf = n - greedy_cover
    
    best_cover = greedy_cover
    
    deadline_check = deadline if deadline else time.time() + 1.0
    
    def _bnb(uncovered, cover_mask, cover_size, depth):
        nonlocal best_cover
        if time.time() > deadline_check:
            return
        if not uncovered:
            if cover_size < best_cover:
                best_cover = cover_size
            return
        if cover_size >= best_cover:
            return
        
        tri = uncovered[0]
        avail = tri & ~cover_mask
        
        vertices_in_tri = []
        m = avail
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            cnt = 0
            for t in uncovered:
                if (t >> v) & 1:
                    cnt += 1
            vertices_in_tri.append((cnt, v))
            m ^= lsb
        vertices_in_tri.sort(reverse=True)
        
        for cnt, v in vertices_in_tri:
            if time.time() > deadline_check:
                return
            new_cover = cover_mask | (1 << v)
            new_uncovered = [t for t in uncovered if not (t & (1 << v))]
            _bnb(new_uncovered, new_cover, cover_size + 1, depth + 1)
            if best_cover == 0:
                return
    
    _bnb(triangles, 0, 0, 0)
    
    return n - best_cover

def _max_triangle_free_subgraph_size(edge_set, n=18, deadline=None):
    adj = _edges_to_adj(edge_set, n)
    return _max_triangle_free_subgraph_size_adj(adj, n, deadline)

def _construct_balanced_tripartite(n=18):
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    return edges

def _try_structured_move(edge_set, adj, n=18, rng=None):
    """
    Try adding an edge within a part of the tripartite structure,
    and then removing edges to fix any K4s created.
    """
    if rng is None:
        rng = random.Random()
        
    # Identify parts (assumes structure is close to tripartite)
    # For now, just try random internal edges
    # Vertices 0-5, 6-11, 12-17
    parts = [(0, 6), (6, 12), (12, 18)]
    
    # Pick a part
    p_idx = rng.randint(0, 2)
    start, end = parts[p_idx]
    
    # Pick two vertices in this part that are not connected
    u, v = rng.sample(range(start, end), 2)
    
    if (u, v) in edge_set or (v, u) in edge_set:
        return None
        
    new_set = set(edge_set)
    new_set.add((u, v))
    new_adj = _edges_to_adj(new_set, n)
    
    # Check for K4s and fix
    # A K4 is formed if there exist w, x such that u-v, u-w, v-w, u-x, v-x, w-x exist
    # We added u-v. So we need to check all w, x.
    # To be safe, we can just run _is_k4_free_adj and if false, try to remove edges.
    # But removing edges is hard.
    # Instead, let's just check if it's K4-free. If not, reject.
    # This move is only valid if adding u-v doesn't create a K4.
    # This happens if there is NO pair (w,x) from other parts that are connected to both u and v and to each other.
    
    if not _is_k4_free_adj(new_adj, n):
        return None
        
    return new_set, new_adj

def _local_search(edges, adj, n=18, max_iters=500, seed=None, deadline=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
        
    edge_set = set(tuple(sorted(e)) for e in edges)
    current_score = _max_triangle_free_subgraph_size(edge_set, n, deadline)
    best_edges = list(edge_set)
    best_score = current_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Try structured move
        result = _try_structured_move(edge_set, adj, n, rng)
        if result:
            new_set, new_adj = result
            new_score = _max_triangle_free_subgraph_size(new_set, n, deadline)
            if new_score < current_score:
                current_score = new_score
                edge_set = new_set
                adj = new_adj
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edge_set)
                    
        # Also try random edge swaps (removal/addition)
        if edge_set:
            rem_edge = rng.choice(list(edge_set))
            all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            cand = rng.choice(all_edges)
            if cand not in edge_set:
                new_set = edge_set - {rem_edge} | {cand}
                new_adj = _edges_to_adj(new_set, n)
                if _is_k4_free_adj(new_adj, n):
                    new_score = _max_triangle_free_subgraph_size(new_set, n, deadline)
                    if new_score < current_score:
                        current_score = new_score
                        edge_set = new_set
                        adj = new_adj
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = list(edge_set)
                            
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    n = 18
    
    base_edges = _construct_balanced_tripartite(n)
    base_set = set(tuple(sorted(e)) for e in base_edges)
    base_adj = _edges_to_adj(base_set, n)
    base_score = _max_triangle_free_subgraph_size(base_set, n, deadline)
    
    best_edges = base_edges
    best_score = base_score
    
    for trial in range(50):
        if time.time() > deadline:
            break
            
        # Start from base or best
        if trial % 5 == 0:
            start_edges = base_edges
        else:
            start_edges = best_edges
            
        start_set = set(tuple(sorted(e)) for e in start_edges)
        start_adj = _edges_to_adj(start_set, n)
        
        ls_edges, ls_score = _local_search(start_edges, start_adj, n, max_iters=100, seed=trial, deadline=deadline)
        
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
            
    return [[u, v] for (u, v) in best_edges]
# EVOLVE-BLOCK-END
