import time
import random
import itertools

_START = time.time()
_DEADLINE = 140.0
_N = 18
_MASK_ALL = (1 << _N) - 1

def _elapsed():
    return time.time() - _START

def _is_k4_free_addition(adj, u, v):
    """Check if adding edge (u,v) keeps graph K4-free."""
    if (adj[u] >> v) & 1:
        return True
    common = adj[u] & adj[v]
    c = common
    while c:
        lsb = c & -c
        x = lsb.bit_length() - 1
        c ^= lsb
        if adj[x] & common:
            return False
    return True

def _build_triangle_adj(adj):
    """Build adjacency list for triangle graph (vertices are triangles, edges if share a vertex)."""
    # Actually, we need: t_adj[u] has bit v set if u and v form a triangle together with some w
    # Wait, let me re-read. The score is largest triangle-free induced subgraph.
    # A set S is triangle-free if no three vertices in S form a triangle.
    # So we need to find the largest subset with no triangles.
    # This is equivalent to: in the "triangle graph" where vertices are original vertices
    # and edges connect pairs that are in some triangle, find the largest independent set.
    
    # Actually simpler: build a graph H where (u,v) is an edge in H iff u,v are in a triangle.
    # Then find MIS of H.
    
    tri_adj = [0] * _N
    for u in range(_N):
        for v in range(u + 1, _N):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                if common:
                    # u and v are in a triangle
                    tri_adj[u] |= 1 << v
                    tri_adj[v] |= 1 << u
    return tri_adj

def _has_triangle_free_subset(tri_adj, size):
    """Check if there exists a triangle-free induced subgraph of given size."""
    # This is MIS on tri_adj of size >= size
    best = [0]
    
    def _solve(mask, size_cur):
        if not mask:
            if size_cur > best[0]:
                best[0] = size_cur
            return
        if size_cur + mask.bit_count() <= best[0]:
            return
        if best[0] >= size:
            return  # Early termination
        
        v = (mask & -mask).bit_length() - 1
        
        # Include v
        new_mask = mask & ~tri_adj[v] & ~(1 << v)
        _solve(new_mask, size_cur + 1)
        
        # Exclude v
        _solve(mask & ~(1 << v), size_cur)
    
    _solve(_MASK_ALL, 0)
    return best[0] >= size

def _max_triangle_free_size(tri_adj):
    """Find the size of the largest triangle-free induced subgraph."""
    best = [0]
    
    def _solve(mask, size_cur):
        if not mask:
            if size_cur > best[0]:
                best[0] = size_cur
            return
        if size_cur + mask.bit_count() <= best[0]:
            return
        
        # Choose vertex with highest degree in remaining graph for better pruning
        max_deg = -1
        chosen_v = -1
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            deg = (tri_adj[x] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                chosen_v = x
            m ^= lsb
        
        v = chosen_v
        
        # Include v
        new_mask = mask & ~tri_adj[v] & ~(1 << v)
        _solve(new_mask, size_cur + 1)
        
        # Exclude v
        _solve(mask & ~(1 << v), size_cur)
    
    _solve(_MASK_ALL, 0)
    return best[0]

def _generate_random_k4free():
    """Generate a random maximal K4-free graph."""
    adj = [0] * _N
    edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    random.shuffle(edges)
    for u, v in edges:
        if _is_k4_free_addition(adj, u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    return adj

def _try_add_edge_to_break_triangle_free(adj, subset):
    """Try to add an edge within subset to break its triangle-freeness, while maintaining K4-free."""
    # Find a triangle-free subset
    tri_adj = _build_triangle_adj(adj)
    
    # For each pair in subset that's not connected, try adding the edge
    for u in subset:
        for v in subset:
            if u >= v:
                continue
            if not ((adj[u] >> v) & 1):
                if _is_k4_free_addition(adj, u, v):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    return True
    return False

def construct():
    deadline = _START + _DEADLINE
    best_score = 18
    best_adj = [0] * _N
    
    random.seed(42)
    
    # Strategy: Generate random K4-free graphs, compute score, keep best
    # Then try to improve by adding edges to break large triangle-free subsets
    
    iterations = 0
    while _elapsed() < deadline - 10:
        adj = _generate_random_k4free()
        tri_adj = _build_triangle_adj(adj)
        score = _max_triangle_free_size(tri_adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
        
        # Try local improvement: if score is large, try to add edges to reduce it
        if score >= 11 and _elapsed() < deadline - 15:
            # Find a large triangle-free subset and try to break it
            # This is expensive, so only do it occasionally
            if iterations % 10 == 0:
                improved = True
                while improved and _elapsed() < deadline - 15:
                    improved = False
                    tri_adj = _build_triangle_adj(adj)
                    score = _max_triangle_free_size(tri_adj)
                    
                    if score >= 11:
                        # Find a triangle-free subset of size score
                        # Use the MIS algorithm but track the actual set
                        best_set = [0]
                        best_size = [0]
                        
                        def _find_mis(mask, size_cur, current_set):
                            if not mask:
                                if size_cur > best_size[0]:
                                    best_size[0] = size_cur
                                    best_set[0] = current_set
                                return
                            if size_cur + mask.bit_count() <= best_size[0]:
                                return
                            
                            max_deg = -1
                            chosen_v = -1
                            m = mask
                            while m:
                                lsb = m & -m
                                x = lsb.bit_length() - 1
                                deg = (tri_adj[x] & mask).bit_count()
                                if deg > max_deg:
                                    max_deg = deg
                                    chosen_v = x
                                m ^= lsb
                            
                            v = chosen_v
                            
                            # Include v
                            new_mask = mask & ~tri_adj[v] & ~(1 << v)
                            _find_mis(new_mask, size_cur + 1, current_set | (1 << v))
                            
                            # Exclude v
                            _find_mis(mask & ~(1 << v), size_cur, current_set)
                        
                        _find_mis(_MASK_ALL, 0, 0)
                        
                        if best_size[0] >= 11:
                            subset_mask = best_set[0]
                            subset = [i for i in range(_N) if (subset_mask >> i) & 1]
                            # Try to add an edge within this subset
                            for u in subset:
                                for v in subset:
                                    if u >= v:
                                        continue
                                    if not ((adj[u] >> v) & 1):
                                        if _is_k4_free_addition(adj, u, v):
                                            adj[u] |= 1 << v
                                            adj[v] |= 1 << u
                                            improved = True
                                            break
                                if improved:
                                    break
        
        iterations += 1
        if iterations > 10000:
            break
    
    # Final check of best_adj
    if best_score == 18:
        # Fallback: use a known good construction
        # Complete bipartite graph K9,9 is K4-free
        best_adj = [0] * _N
        for u in range(9):
            for v in range(9, 18):
                best_adj[u] |= 1 << v
                best_adj[v] |= 1 << u
        best_score = _max_triangle_free_size(_build_triangle_adj(best_adj))
    
    # Convert best_adj to edges
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
