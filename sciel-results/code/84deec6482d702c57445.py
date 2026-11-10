import itertools
import random
import time

def _is_non_2_colorable_fast(n, edges):
    """
    Fast check for non-2-colorability using bitmask backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable (i.e., no valid coloring exists).
    """
    if not edges:
        return False

    # Precompute edge bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    full_mask = (1 << n) - 1
    
    # Precompute vertex degrees for heuristic ordering
    # But since n is small, we can just pick the lowest unassigned vertex
    # or a vertex involved in the most "tight" edges.
    # Let's stick to a simple, robust approach first.
    
    def propagate(color1, unassigned):
        """
        Apply unit propagation. Returns (new_color1, new_unassigned, conflict_flag).
        """
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                # Case 1: All 5 assigned
                if unassigned_in_edge == 0:
                    # Check if monochromatic
                    c1_count = bin(assigned_in_edge & color1).count('1')
                    if c1_count == 0 or c1_count == 5:
                        return None, None, True
                    continue
                
                # Case 2: Exactly 1 unassigned
                if unassigned_in_edge & (unassigned_in_edge - 1) == 0:
                    # It's a single bit
                    other_4 = mask ^ unassigned_in_edge
                    c1_count = bin(other_4 & color1).count('1')
                    if c1_count == 4:
                        # Force unassigned to color 0
                        # Just remove from unassigned
                        unassigned ^= unassigned_in_edge
                        changed = True
                    elif c1_count == 0:
                        # Force unassigned to color 1
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True
                        
        return color1, unassigned, False

    def backtrack(color1, unassigned):
        # Check for immediate conflict via propagation
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False
        
        if unassigned == 0:
            # All assigned. Check if any edge is monochromatic.
            # Propagation should have caught this, but double check.
            for mask in edge_masks:
                c1_count = bin(mask & color1).count('1')
                if c1_count == 0 or c1_count == 5:
                    return False
            return True # Valid coloring found
            
        # Pick a vertex to assign.
        # Heuristic: Pick the vertex with the highest number of edges where it is the only unassigned one?
        # Or just the lowest bit.
        # Let's use a simple heuristic: pick the vertex that appears in the most edges with exactly 1 unassigned vertex?
        # That might be expensive to compute each step.
        # Let's just pick the lowest bit for now.
        v_bit = unassigned & -unassigned
        
        # Try color 1
        if backtrack(color1 | v_bit, unassigned ^ v_bit):
            return True
            
        # Try color 0
        if backtrack(color1, unassigned ^ v_bit):
            return True
            
        return False

    # Symmetry breaking: Fix vertex 0 to color 0.
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1
    
    # Propagate initial state
    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True
        
    if initial_unassigned == 0:
        for mask in edge_masks:
            c1_count = bin(mask & initial_color1).count('1')
            if c1_count == 0 or c1_count == 5:
                return False
        return False
        
    if backtrack(initial_color1, initial_unassigned):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin

    # Strategy:
    # 1. Start with a known non-2-colorable hypergraph.
    #    K9^(5) is one, but has 126 edges.
    #    We can try to find a smaller one.
    #    The best known is 51 edges.
    #    We can start with a random subset of K9^(5) or a specific construction.
    #    Let's start with a random subset of size ~60 and prune.
    #    Or better, start with K9^(5) and prune aggressively.
    
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Randomize the order of edges to avoid bias in pruning
    random.shuffle(all_edges)
    
    # Start with a subset?
    # If we start with all 126, pruning takes time.
    # If we start with too few, it might be 2-colorable.
    # Let's start with all 126, but use a faster pruning loop.
    
    current_edges = all_edges[:]
    
    # Optimization:
    # Instead of checking _is_non_2_colorable_fast for every single removal attempt,
    # we can try to remove multiple edges at once? No, that's risky.
    # But we can cache the result of the check for the current set.
    # Actually, the check is fast enough for n=9.
    
    # Let's try a different approach:
    # Randomly generate subsets of edges of size k, and check if they are non-2-colorable.
    # Decrease k until we find the minimum.
    # This is a binary search on the number of edges.
    
    # However, finding a specific subset of size k that is non-2-colorable is a search problem.
    # Pruning from a larger set is more reliable.
    
    # Let's stick to pruning, but make it smarter.
    # We can try to remove edges in a random order, and if a removal fails, we skip it for now.
    # We loop until no more edges can be removed.
    
    # To speed up, we can use a "soft" constraint:
    # If removing an edge makes the hypergraph 2-colorable, we keep it.
    # Otherwise, we remove it.
    
    # Let's implement the pruning loop.
    
    removed_any = True
    while removed_any and (time.time() - start_time) < deadline:
        removed_any = False
        # Shuffle to try different orders
        random.shuffle(current_edges)
        
        # We iterate from end to start to allow removal
        for i in range(len(current_edges) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
                
            # Try removing current_edges[i]
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if new_edges is non-2-colorable
            if _is_non_2_colorable_fast(n, new_edges):
                current_edges = new_edges
                removed_any = True
            # Else, keep it
    
    # If we still have many edges, we might try to restart with a different random seed or subset.
    # But for now, this should get us close to the minimum.
    
    # Final verification
    if not _is_non_2_colorable_fast(n, current_edges):
        # Fallback
        return {'n': n, 'edges': all_edges}
        
    return {'n': n, 'edges': current_edges}

# To improve further, we can try to find a construction with fewer vertices?
# No, n must be at least 9 for 5-uniform non-2-colorable (Property B).
# Actually, m(5) is the minimum number of edges.
# The vertices can be more than 9, but usually small n is better for dense hypergraphs.
# However, sparse hypergraphs with more vertices might have fewer edges?
# Unlikely. The density is high.

# Let's add a timeout guard to the pruning loop to ensure we return within 120s.
# The current loop has a deadline check.

# One more idea:
# Use a SAT solver approach? No, standard library only.
# Bitmask backtracking is the way to go.

# Let's refine the vertex selection heuristic in backtrack.
# Instead of lowest bit, pick the vertex that is in the most "critical" edges.
# An edge is critical if it has 1 unassigned vertex.
# If we assign that vertex, we might resolve the edge.
# This is a form of "fail-first".

# Let's modify backtrack to use this heuristic.

def _is_non_2_colorable_fast_v2(n, edges):
    if not edges:
        return False

    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    full_mask = (1 << n) - 1
    
    def propagate(color1, unassigned):
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                if unassigned_in_edge == 0:
                    c1_count = bin(assigned_in_edge & color1).count('1')
                    if c1_count == 0 or c1_count == 5:
                        return None, None, True
                    continue
                
                if unassigned_in_edge & (unassigned_in_edge - 1) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    c1_count = bin(other_4 & color1).count('1')
                    if c1_count == 4:
                        unassigned ^= unassigned_in_edge
                        changed = True
                    elif c1_count == 0:
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True
                        
        return color1, unassigned, False

    def get_heuristic_vertex(color1, unassigned):
        # Find the vertex that appears in the most edges with exactly 1 unassigned vertex
        # Or just the lowest bit if none.
        counts = [0] * n
        for mask in edge_masks:
            unassigned_in_edge = mask & unassigned
            if unassigned_in_edge & (unassigned_in_edge - 1) == 0:
                # Single unassigned vertex
                v = unassigned_in_edge.bit_length() - 1
                counts[v] += 1
        
        best_v = -1
        best_count = 0
        for v in range(n):
            if unassigned & (1 << v):
                if counts[v] > best_count:
                    best_count = counts[v]
                    best_v = v
        
        if best_v == -1:
            # No critical edges, pick lowest unassigned
            return unassigned & -unassigned
        else:
            return 1 << best_v

    def backtrack(color1, unassigned):
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False
        
        if unassigned == 0:
            for mask in edge_masks:
                c1_count = bin(mask & color1).count('1')
                if c1_count == 0 or c1_count == 5:
                    return False
            return True
            
        v_bit = get_heuristic_vertex(color1, unassigned)
        
        if backtrack(color1 | v_bit, unassigned ^ v_bit):
            return True
            
        if backtrack(color1, unassigned ^ v_bit):
            return True
            
        return False

    initial_color1 = 0
    initial_unassigned = full_mask ^ 1
    
    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True
        
    if initial_unassigned == 0:
        for mask in edge_masks:
            c1_count = bin(mask & initial_color1).count('1')
            if c1_count == 0 or c1_count == 5:
                return False
        return False
        
    if backtrack(initial_color1, initial_unassigned):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    random.shuffle(all_edges)
    
    current_edges = all_edges[:]
    
    removed_any = True
    while removed_any and (time.time() - start_time) < deadline:
        removed_any = False
        random.shuffle(current_edges)
        
        for i in range(len(current_edges) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
                
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            if _is_non_2_colorable_fast_v2(n, new_edges):
                current_edges = new_edges
                removed_any = True
    
    if not _is_non_2_colorable_fast_v2(n, current_edges):
        return {'n': n, 'edges': all_edges}
        
    return {'n': n, 'edges': current_edges}
