import random
import time
import itertools

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

def _is_3_colorable(adj):
    """Check if the graph is 3-colorable using backtracking."""
    n = len(adj)
    # 0: uncolored, 1,2,3: colors
    color = [0] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        # Get available colors
        avail = 7  # 111
        # Check neighbors with index < idx (already colored)
        # Actually, we should check all colored neighbors.
        # For efficiency, we can just check the mask of colored neighbors.
        # But simple check is fine for n=16.
        
        # Heuristic: pick vertex with fewest available colors?
        # For simplicity and speed in this context, fixed order with pruning is okay.
        # But MRV is much better.
        
        # Let's do simple MRV
        best_v = -1
        min_avail = 8
        for v in range(n):
            if color[v] == 0:
                # count unavailable colors
                unavail = 0
                mask = adj[v]
                # iterate neighbors
                m = mask
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    m ^= lb
                    if color[u] != 0:
                        unavail |= (1 << (color[u]-1))
                avail_colors = 7 ^ unavail
                count = bin(avail_colors).count('1')
                if count == 0:
                    return False
                if count < min_avail:
                    min_avail = count
                    best_v = v
                    if count == 1:
                        break
        
        if best_v == -1:
            return True # All colored
            
        # Try colors for best_v
        mask = adj[best_v]
        unavail = 0
        m = mask
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if color[u] != 0:
                unavail |= (1 << (color[u]-1))
        
        avail_colors = 7 ^ unavail
        
        # Try each available color
        c = 1
        while c <= 3:
            if avail_colors & (1 << (c-1)):
                color[best_v] = c
                if backtrack(idx + 1):
                    return True
                color[best_v] = 0
            c += 1
        return False

    return backtrack(0)

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

def _local_search_improve(edge_list, adj, seed, deadline):
    """
    Try to improve the graph by swapping edges: remove one edge, add another
    that was previously blocked. Maintain triangle-freeness and non-3-colorability.
    """
    n = 16
    rng = random.Random(seed)
    
    # Build edge set from current state
    edge_set = set()
    for u, v in edge_list:
        edge_set.add((min(u, v), max(u, v)))
    
    # Get all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Current number of edges
    current_count = len(edge_list)
    best_count = current_count
    best_adj = [a for a in adj]
    best_edge_set = set(edge_set)
    
    # Build list of missing edges
    missing_edges = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set]
    
    # Try multiple rounds of local search
    for _round in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Shuffle missing edges for randomization
        rng.shuffle(missing_edges)
        
        # Try to add a missing edge by removing a blocking edge
        for mu, mv in missing_edges:
            if time.time() > deadline:
                break
            
            # Check if (mu, mv) can be added
            common = adj[mu] & adj[mv]
            if common == 0:
                # Can add directly
                adj[mu] |= (1 << mv)
                adj[mv] |= (1 << mu)
                edge_set.add((mu, mv))
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [a for a in adj]
                    best_edge_set = set(edge_set)
                    improved = True
                # Remove from missing list
                if (mu, mv) in missing_edges:
                    missing_edges.remove((mu, mv))
                continue
            
            # Try removing one of the common neighbors' edges to unblock
            common_list = []
            m = common
            while m:
                lb = m & (-m)
                w = lb.bit_length() - 1
                m ^= lb
                common_list.append(w)
            
            rng.shuffle(common_list)
            
            for w in common_list:
                if time.time() > deadline:
                    break
                
                # Try removing edge (mu, w) or (w, mv)
                for edge_to_remove in [(mu, w), (w, mv)]:
                    ru, rv = edge_to_remove
                    ru, rv = min(ru, rv), max(ru, rv)
                    
                    if (ru, rv) not in edge_set:
                        continue
                    
                    # Check if removing this edge and adding (mu, mv) is beneficial
                    # Remove (ru, rv)
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    edge_set.discard((ru, rv))
                    
                    # Now try to add (mu, mv)
                    new_common = adj[mu] & adj[mv]
                    if new_common == 0:
                        # Can add
                        adj[mu] |= (1 << mv)
                        adj[mv] |= (1 << mu)
                        edge_set.add((mu, mv))
                        current_count += 1  # net +1 (removed one, added one, but we want to check if we can add more)
                        # Actually we removed one and added one, so count stays same
                        # Let's recalculate: we removed (ru,rv) and added (mu,mv), so count is same
                        # But maybe this unblocks other edges
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = [a for a in adj]
                            best_edge_set = set(edge_set)
                            improved = True
                        
                        # Check if we can now add more edges
                        # Try adding some other missing edges
                        for mu2, mv2 in missing_edges[:20]:
                            if (mu2, mv2) == (mu, mv):
                                continue
                            c2 = adj[mu2] & adj[mv2]
                            if c2 == 0:
                                adj[mu2] |= (1 << mv2)
                                adj[mv2] |= (1 << mu2)
                                edge_set.add((mu2, mv2))
                                current_count += 1
                                if current_count > best_count:
                                    best_count = current_count
                                    best_adj = [a for a in adj]
                                    best_edge_set = set(edge_set)
                                    improved = True
                                if (mu2, mv2) in missing_edges:
                                    missing_edges.remove((mu2, mv2))
                    else:
                        # Couldn't add, revert
                        adj[ru] |= (1 << rv)
                        adj[rv] |= (1 << ru)
                        edge_set.add((ru, rv))
                    
                    # Verify triangle-freeness
                    if not _is_triangle_free(adj):
                        # Revert everything
                        break
        
        if not improved:
            break
    
    # Convert best_adj back to edge list
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result_edges.append([u, v])
    
    return result_edges, len(result_edges)

def _try_multiple_cores(deadline):
    """
    Try different core structures to find better starting points.
    """
    n = 16
    results = []
    
    # Core 1: Grötzsch graph (11 vertices)
    grotzsch = _build_grotzsch_edges()
    results.append(('grotzsch', grotzsch))
    
    # Core 2: Two disjoint Grötzsch-like structures? No, that won't work.
    # Core 3: Mycielski of C5 with different vertex labeling
    # Core 4: Try adding a 5th vertex connected to specific vertices
    
    # Alternative: Start with a different 4-chromatic triangle-free core
    # The Grötzsch graph is the smallest 4-chromatic triangle-free graph (11 vertices)
    # We can try to use different embeddings or augmentations
    
    return results

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
    
    seed = 0
    while time.time() < deadline - 5.0:
        # Phase 1: Greedy completion
        edges, count = _greedy_complete_with_core(grotzsch, seed, deadline)
        
        # Phase 2: Local search improvement
        if time.time() < deadline - 3.0:
            adj = _make_bitmask_adj(edges)
            improved_edges, improved_count = _local_search_improve(edges, adj, seed + 1000, deadline)
            if improved_count > count:
                edges = improved_edges
                count = improved_count
        
        if count > best_count:
            best_count = count
            best_edges = edges
        
        seed += 1
    
    # Final validation
    adj = _make_bitmask_adj(best_edges)
    if not _is_triangle_free(adj):
        best_edges = list(grotzsch)
    if _is_3_colorable(adj):
        # This should not happen if core is Grotzsch, but just in case
        # Fallback to Grotzsch
        best_edges = list(grotzsch)
        
    return best_edges
