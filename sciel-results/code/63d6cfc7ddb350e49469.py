"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if not (au >> v) & 1:
                continue
            common = au & adj[v]
            for w in range(v + 1, n):
                if (common >> w) & 1:
                    if adj[w] & (common & ~(1 << w)):
                        return False
    return True

def _is_k4_free(edge_set, n=18):
    adj = _edges_to_adj(edge_set, n)
    return _is_k4_free_adj(adj, n)

def _count_triangles_in_mask(mask, adj, n=18):
    """Count number of triangles fully within the vertex set given by mask."""
    cnt = 0
    for u in range(n):
        if not (mask >> u) & 1:
            continue
        au = adj[u] & mask & ~(1 << u)
        for v in range(u + 1, n):
            if not (au >> v) & 1:
                continue
            common = au & adj[v] & mask
            cnt += bin(common).count("1")
    return cnt

def _max_triangle_free_subgraph_size(edge_set, n=18, time_budget=3.0):
    adj = _edges_to_adj(edge_set, n)
    full = (1 << n) - 1
    
    # Precompute for each vertex, the mask of its neighbors
    # Greedy lower bound: repeatedly remove vertex in most triangles
    remaining = full
    while True:
        best_v = -1
        best_c = 0
        for v in range(n):
            if not (remaining >> v) & 1:
                continue
            # Count triangles containing v within remaining
            c = 0
            for w in range(v + 1, n):
                if not (remaining >> w) & 1:
                    continue
                if not (adj[v] >> w) & 1:
                    continue
                common = adj[v] & adj[w] & remaining & ~(1 << v) & ~(1 << w)
                c += bin(common).count("1")
            if c > best_c:
                best_c = c
                best_v = v
        if best_c == 0:
            break
        remaining &= ~(1 << best_v)
    
    greedy_tf_size = bin(remaining).count("1")
    
    # If no triangles at all, return n
    if _count_triangles_in_mask(full, adj, n) == 0:
        return n
    
    # Exact B&B: find maximum triangle-free induced subgraph
    # State: candidate_mask (vertices we're considering), 
    # We search for the largest subset of candidate_mask that is triangle-free.
    # Key insight: if candidate_mask contains no triangles, its size is a valid answer.
    # Branch: pick a vertex v in candidate_mask that is in the most triangles within candidate_mask.
    #   Branch 1: exclude v -> solve(candidate_mask \ {v})
    #   Branch 2: include v -> then all vertices that form a triangle with v must be excluded.
    #              solve(candidate_mask \ {v} \ {vertices forming triangle with v})
    
    deadline = time.time() + time_budget
    best_size = [greedy_tf_size]
    
    # Memoize: for a given mask, if we've already determined it's triangle-free, cache its size
    # But masks are 2^18 = 262144 possible, so we can use a dict
    
    def _is_triangle_free(mask):
        return _count_triangles_in_mask(mask, adj, n) == 0
    
    def _bnb(mask):
        if time.time() > deadline:
            return
        if mask == 0:
            return
        sz = bin(mask).count("1")
        if sz <= best_size[0]:
            return
        # Check if mask is already triangle-free
        if _is_triangle_free(mask):
            if sz > best_size[0]:
                best_size[0] = sz
            return
        # Find vertex in mask that is in the most triangles within mask
        best_v = -1
        best_tc = 0
        for v in range(n):
            if not (mask >> v) & 1:
                continue
            tc = 0
            for w in range(v + 1, n):
                if not (mask >> w) & 1:
                    continue
                if not (adj[v] >> w) & 1:
                    continue
                common = adj[v] & adj[w] & mask & ~(1 << v) & ~(1 << w)
                tc += bin(common).count("1")
            if tc > best_tc:
                best_tc = tc
                best_v = v
        
        if best_v < 0 or best_tc == 0:
            # Shouldn't happen if mask has triangles, but safety
            if sz > best_size[0]:
                best_size[0] = sz
            return
        
        # Branch 1: exclude best_v
        _bnb(mask & ~(1 << best_v))
        
        # Branch 2: include best_v -> exclude all vertices that form a triangle with best_v
        # Vertices forming a triangle with best_v: for each edge (best_v, w) in mask,
        # any common neighbor x in mask forms triangle (best_v, w, x)
        exclude_mask = 1 << best_v
        for w in range(n):
            if not (mask >> w) & 1 or w == best_v:
                continue
            if not (adj[best_v] >> w) & 1:
                continue
            # w is adjacent to best_v; find common neighbors in mask
            common = adj[best_v] & adj[w] & mask & ~(1 << best_v) & ~(1 << w)
            exclude_mask |= common
        # Also exclude best_v itself from the "include" branch's remaining
        new_mask = mask & ~exclude_mask
        _bnb(new_mask)
    
    _bnb(full)
    return best_size[0]

def _random_greedy_k4_free(n=18, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    adj = [0] * n
    edge_set = set()
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        common = adj[u] & adj[v]
        ok = True
        cm = common
        while cm:
            lsb = cm & (-cm)
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~(1 << w)):
                ok = False
                break
            cm ^= lsb
        if ok:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
    
    return list(edge_set)

def _local_search(edges, n=18, max_iters=500, seed=None, deadline=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _edges_to_adj(edge_set, n)
    current_score = _max_triangle_free_subgraph_size(edge_set, n, time_budget=1.5)
    best_edges = list(edge_set)
    best_score = current_score
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try a random edge swap: remove one edge, add one
        if len(edge_set) > 0:
            rem_edge = rng.choice(list(edge_set))
        else:
            break
        
        cand = rng.choice(all_edges)
        
        if cand == rem_edge:
            continue
        
        new_set = edge_set - {rem_edge} | {cand}
        new_adj = _edges_to_adj(new_set, n)
        
        if not _is_k4_free_adj(new_adj, n):
            continue
        
        new_score = _max_triangle_free_subgraph_size(new_set, n, time_budget=1.5)
        
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
    deadline = start_time + 145  # safety margin before 150s kill
    
    n = 18
    
    # Start with the known good construction: complete tripartite K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    base_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    base_score = _max_triangle_free_subgraph_size(base_edges, n, time_budget=2.0)
    
    best_edges = base_edges
    best_score = base_score
    
    # Try random constructions + local search
    for trial in range(60):
        if time.time() > deadline:
            break
        
        # Build random K4-free graph
        edges = _random_greedy_k4_free(n, seed=trial)
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Check it's valid
        if not _is_k4_free(edge_set, n):
            continue
        
        score = _max_triangle_free_subgraph_size(edge_set, n, time_budget=1.5)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        # Local search with tighter deadline
        ls_deadline = min(deadline, start_time + 146)
        ls_edges, ls_score = _local_search(edges, n, max_iters=100, seed=trial, deadline=ls_deadline)
        
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
        
        # Also try local search on the base
        if trial == 0:
            ls_base, ls_base_score = _local_search(base_edges, n, max_iters=200, seed=999, deadline=ls_deadline)
            if ls_base_score < best_score:
                best_score = ls_base_score
                best_edges = ls_base
    
    return [[u, v] for (u, v) in best_edges]
# EVOLVE-BLOCK-END
