import random
import time

def _build_grotzsch_edges():
    """Return the edge list of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    # Cycle on 0..4
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    # Mycielski: vertices 5..9 are clones of 0..4
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    # Vertex 10 connected to all clones 5..9
    edges += [[10, 5 + u] for u in range(5)]
    return edges

def _make_bitmask_adj(edge_list, n=16):
    """Convert edge list to list of integer bitmasks representing adjacency."""
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_triangle_free(adj):
    """Verify that adjacency bitmasks define a simple triangle-free graph."""
    n = len(adj)
    for u in range(n):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

def _is_3_colorable(adj, deadline=None):
    """Check if the graph is 3-colorable using backtracking with MRV heuristic."""
    n = len(adj)
    color = [0] * n
    
    def backtrack():
        if deadline and time.time() > deadline:
            return None  # timeout
        
        # MRV: pick uncolored vertex with fewest available colors
        best_v = -1
        min_avail = 4
        best_avail_colors = 0
        
        for v in range(n):
            if color[v] == 0:
                unavail = 0
                mask = adj[v]
                while mask:
                    lb = mask & (-mask)
                    u = lb.bit_length() - 1
                    mask ^= lb
                    if color[u] != 0:
                        unavail |= (1 << (color[u]-1))
                avail_colors = 7 ^ unavail
                count = bin(avail_colors).count('1')
                if count == 0:
                    return False
                if count < min_avail:
                    min_avail = count
                    best_v = v
                    best_avail_colors = avail_colors
                    if count == 1:
                        break
        
        if best_v == -1:
            return True  # All colored
            
        # Try each available color for best_v
        c = 1
        while c <= 3:
            if best_avail_colors & (1 << (c-1)):
                color[best_v] = c
                result = backtrack()
                if result is not None:
                    return result
                color[best_v] = 0
            c += 1
        return False

    result = backtrack()
    return result if result is not None else True  # On timeout, assume colorable (conservative)

def _greedy_complete_with_core(core_edges, seed, deadline):
    """
    Start from core edges, greedily add edges in random order while maintaining
    triangle-freeness.
    """
    n = 16
    rng = random.Random(seed)
    
    adj = _make_bitmask_adj(core_edges, n)
    edge_set = set()
    for u, v in core_edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # All possible edges not in core
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    remaining = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set]
    rng.shuffle(remaining)
    
    # Greedy addition
    for u, v in remaining:
        if time.time() > deadline:
            break
        # Check if adding (u, v) creates a triangle
        common = adj[u] & adj[v]
        if common == 0:
            # Safe to add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            
    # Convert back to edge list
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                result_edges.append([u, v])
                
    return result_edges, len(result_edges)

def _local_search_improve(edges, seed, deadline):
    """
    Try to improve the edge count via local search:
    - Remove an edge, then greedily re-add edges to see if we can get more.
    - Or try swapping: remove one edge, add another if it maintains triangle-freeness.
    """
    n = 16
    rng = random.Random(seed)
    
    adj = _make_bitmask_adj(edges, n)
    current_edges = list(edges)
    current_count = len(current_edges)
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    # Build list of possible edges
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each edge and then greedily re-adding
        for idx in range(len(current_edges)):
            if time.time() > deadline:
                break
            u, v = current_edges[idx]
            # Temporarily remove edge (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edge_set.discard((min(u,v), max(u,v)))
            
            # Try to add other edges greedily
            added_any = False
            for eu, ev in all_possible:
                if (min(eu,ev), max(eu,ev)) in edge_set:
                    continue
                common = adj[eu] & adj[ev]
                if common == 0:
                    adj[eu] |= (1 << ev)
                    adj[ev] |= (1 << eu)
                    edge_set.add((min(eu,ev), max(eu,ev)))
                    added_any = True
            
            # Count edges now
            new_count = len(edge_set)
            if new_count > current_count:
                # Keep this improvement
                current_count = new_count
                current_edges = [[u, v] for (u, v) in edge_set if u < v]
                improved = True
                break
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((min(u,v), max(u,v)))
                
    return current_edges, current_count

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    grotzsch = _build_grotzsch_edges()
    best_edges = list(grotzsch)
    best_count = len(grotzsch)
    
    # Phase 1: Greedy completion with multiple seeds
    seed = 0
    while time.time() < deadline - 30.0:
        edges, count = _greedy_complete_with_core(grotzsch, seed, deadline - 30.0)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        # Also try local search on the best found so far occasionally
        if seed % 10 == 0 and time.time() < deadline - 10.0:
            ls_edges, ls_count = _local_search_improve(best_edges, seed, deadline - 5.0)
            if ls_count > best_count:
                best_count = ls_count
                best_edges = ls_edges
    
    # Phase 2: Final local search polish
    if time.time() < deadline - 2.0:
        ls_edges, ls_count = _local_search_improve(best_edges, 999, deadline - 1.0)
        if ls_count > best_count:
            best_count = ls_count
            best_edges = ls_edges
    
    # Final validation
    adj = _make_bitmask_adj(best_edges)
    if not _is_triangle_free(adj):
        best_edges = list(grotzsch)
        best_count = len(grotzsch)
    
    # Verify non-3-colorability (use a short deadline for this check)
    colorable = _is_3_colorable(adj, deadline=start_time + 115.0)
    if colorable:
        # Fallback to Grotzsch which is known to be non-3-colorable
        best_edges = list(grotzsch)
        best_count = len(grotzsch)
        
    return best_edges
