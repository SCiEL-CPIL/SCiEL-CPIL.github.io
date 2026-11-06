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
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                    return False
    return True

def _is_3_colorable_fast(adj, n=16):
    """
    Fast 3-colorability check using bitmask-based constraint propagation.
    Returns True if 3-colorable, False otherwise.
    Uses DSATUR-like heuristic with bitmask propagation.
    """
    # color_mask[v] = bitmask of colors still available for vertex v
    # Initially all 3 colors available: 0b111 = 7
    # When we assign color c to vertex v, we remove color c from all neighbors.
    
    # We'll use a stack-based backtracking with MRV heuristic.
    # State: list of (vertex, assigned_color) and the current color_masks.
    
    # For speed, we'll use iterative deepening or just careful backtracking.
    
    # Precompute neighbor lists as bitmasks for quick access
    # adj is already bitmask adjacency
    
    # Use a recursive approach with early termination
    # color[v] = 0 means uncolored, 1-3 means assigned color
    color = [0] * n
    
    # Precompute for each vertex, the list of neighbors
    # Actually, adj[v] already gives us the bitmask of neighbors
    
    def get_avail_colors(v):
        """Get bitmask of available colors for vertex v given current coloring."""
        avail = 7  # all 3 colors
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if color[u] != 0:
                avail &= ~(1 << (color[u] - 1))
        return avail
    
    def backtrack():
        # Find uncolored vertex with fewest available colors (MRV)
        best_v = -1
        best_avail = 8
        for v in range(n):
            if color[v] == 0:
                avail = get_avail_colors(v)
                cnt = bin(avail).count('1')
                if cnt == 0:
                    return False  # Dead end
                if cnt < best_avail:
                    best_avail = cnt
                    best_v = v
                    if cnt == 1:
                        break
        
        if best_v == -1:
            return True  # All colored
        
        avail = get_avail_colors(best_v)
        
        # Try each available color
        for c in range(1, 4):
            if avail & (1 << (c - 1)):
                color[best_v] = c
                if backtrack():
                    return True
                color[best_v] = 0
        
        return False
    
    return backtrack()

def _greedy_complete_with_core(core_edges, seed, deadline, strategy=0):
    """
    Start from core edges, greedily add edges while maintaining triangle-freeness.
    Different strategies for edge ordering.
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
    
    # Different strategies for ordering
    if strategy == 0:
        # Pure random
        rng.shuffle(remaining)
    elif strategy == 1:
        # Sort by sum of degrees (prefer edges between low-degree vertices)
        deg = [bin(adj[u]).count('1') for u in range(n)]
        remaining.sort(key=lambda e: deg[e[0]] + deg[e[1]])
        # Add some randomness
        for i in range(len(remaining) - 1):
            j = rng.randint(i, len(remaining) - 1)
            remaining[i], remaining[j] = remaining[j], remaining[i]
    elif strategy == 2:
        # Sort by product of degrees
        deg = [bin(adj[u]).count('1') for u in range(n)]
        remaining.sort(key=lambda e: deg[e[0]] * deg[e[1]])
        for i in range(len(remaining) - 1):
            j = rng.randint(i, len(remaining) - 1)
            remaining[i], remaining[j] = remaining[j], remaining[i]
    elif strategy == 3:
        # Weighted random: prefer edges where both endpoints have low degree
        deg = [bin(adj[u]).count('1') for u in range(n)]
        # Assign weights inversely proportional to degree sum
        weights = []
        for u, v in remaining:
            w = 1.0 / (1 + deg[u] + deg[v])
            weights.append(w)
        # Weighted shuffle
        for i in range(len(remaining) - 1):
            total_w = sum(weights[i:])
            r = rng.random() * total_w
            cum = 0
            for j in range(i, len(remaining)):
                cum += weights[j]
                if cum >= r:
                    remaining[i], remaining[j] = remaining[j], remaining[i]
                    weights[i], weights[j] = weights[j], weights[i]
                    break
    else:
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
            
    # Local improvement: try to add more edges by reconsidering
    # Do a second pass with different ordering
    remaining2 = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set]
    if remaining2:
        rng2 = random.Random(seed + 1000)
        rng2.shuffle(remaining2)
        for u, v in remaining2:
            if time.time() > deadline:
                break
            common = adj[u] & adj[v]
            if common == 0:
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

def _local_improve(edges, seed, deadline):
    """
    Try to improve the edge count by removing some edges and re-adding.
    This is a simple local search.
    """
    n = 16
    rng = random.Random(seed)
    
    adj = _make_bitmask_adj(edges, n)
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    best_edges = list(edges)
    best_count = len(edges)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Try several rounds of local moves
    for iteration in range(20):
        if time.time() > deadline:
            break
        
        # Pick a random edge to remove
        if not edge_set:
            break
        to_remove = rng.choice(list(edge_set))
        u, v = to_remove
        
        # Remove it
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.discard(to_remove)
        
        # Try to add other edges
        candidates = [(a, b) for a, b in all_edges if (a, b) not in edge_set]
        rng.shuffle(candidates)
        
        added = 0
        for a, b in candidates:
            if time.time() > deadline:
                break
            common = adj[a] & adj[b]
            if common == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                edge_set.add((a, b))
                added += 1
        
        # If we gained edges, keep this state
        current_count = len(edge_set)
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v] for u, v in sorted(edge_set)]
        else:
            # Revert: we need to restore the original state
            # This is complex, so let's just continue with the modified state
            # Actually, let's keep the modified state as it might lead to better solutions
            pass
    
    return best_edges, best_count

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    grotzsch = _build_grotzsch_edges()
    best_edges = list(grotzsch)
    best_count = len(grotzsch)
    
    seed = 0
    strategies = [0, 1, 2, 3]
    
    while time.time() < deadline - 3.0:
        strategy = strategies[seed % len(strategies)]
        edges, count = _greedy_complete_with_core(grotzsch, seed, deadline, strategy)
        
        # Validate: must be triangle-free and not 3-colorable
        adj = _make_bitmask_adj(edges)
        if _is_triangle_free(adj) and not _is_3_colorable_fast(adj):
            if count > best_count:
                best_count = count
                best_edges = edges
                # Try local improvement
                if time.time() < deadline - 5.0:
                    improved_edges, improved_count = _local_improve(edges, seed, deadline)
                    if improved_count > best_count:
                        adj2 = _make_bitmask_adj(improved_edges)
                        if _is_triangle_free(adj2) and not _is_3_colorable_fast(adj2):
                            best_count = improved_count
                            best_edges = improved_edges
        seed += 1
    
    # Final validation
    adj = _make_bitmask_adj(best_edges)
    if not _is_triangle_free(adj) or _is_3_colorable_fast(adj):
        best_edges = list(grotzsch)
        
    return best_edges
