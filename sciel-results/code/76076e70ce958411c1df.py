import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable.
    Returns True if colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Precompute vertex masks for each edge
    # edge_mask[i] is a bitmask of vertices in edge i
    edge_masks = []
    for e in edges:
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks.append(m)
        
    # Precompute for each vertex, which edges it belongs to? 
    # Or just iterate edges in propagation.
    
    # We maintain:
    # unassigned: bitmask of unassigned vertices
    # color0: bitmask of vertices assigned 0
    # color1: bitmask of vertices assigned 1
    
    # Initial state
    unassigned = (1 << n) - 1
    color0 = 0
    color1 = 0
    
    # We'll use a stack for backtracking.
    # State: (unassigned, color0, color1, assignment_history)
    # To save memory and time, we can just recurse or use a stack of states.
    
    def propagate(unassigned, color0, color1):
        # Returns (success, unassigned, color0, color1, assignment)
        # assignment is a list of (vertex, color) forced by propagation in this step
        # Actually, we just need to check for contradictions and force assignments.
        
        forced = []
        changed = True
        while changed:
            changed = False
            for m in edge_masks:
                # Count bits in intersection
                c0 = (m & color0).bit_count()
                c1 = (m & color1).bit_count()
                cu = (m & unassigned).bit_count()
                
                if c0 == 5:
                    return False, None, None, None
                if c1 == 5:
                    return False, None, None, None
                
                if c0 == 4 and cu == 1:
                    # Force the unassigned vertex to 1
                    v_mask = m & unassigned
                    v = v_mask.bit_length() - 1
                    if color1 & v_mask:
                        # Already assigned? Shouldn't happen if unassigned is correct
                        pass
                    # Check if it's already 1
                    if (color1 & v_mask) == 0:
                        color1 |= v_mask
                        unassigned &= ~v_mask
                        forced.append((v, 1))
                        changed = True
                elif c1 == 4 and cu == 1:
                    # Force the unassigned vertex to 0
                    v_mask = m & unassigned
                    v = v_mask.bit_length() - 1
                    if (color0 & v_mask) == 0:
                        color0 |= v_mask
                        unassigned &= ~v_mask
                        forced.append((v, 0))
                        changed = True
        return True, unassigned, color0, color1, forced

    def backtrack(unassigned, color0, color1):
        # Check for trivial success: if unassigned is 0, we are done
        # But we must ensure no edge is monochromatic. 
        # The propagation step above handles contradictions (5 of one color).
        # If we reach a state with no contradictions and unassigned is empty, it's a valid coloring.
        # Wait, if unassigned is not empty, we must pick a vertex.
        
        # Run propagation
        ok, unassigned, color0, color1, _ = propagate(unassigned, color0, color1)
        if not ok:
            return False
        
        if unassigned == 0:
            return True
            
        # Pick a vertex to assign
        # Heuristic: pick vertex in the most constrained edge? 
        # For simplicity, pick the lowest set bit.
        v = (unassigned & -unassigned).bit_length() - 1
        
        # Try color 0
        new_unassigned = unassigned & ~(1 << v)
        new_color0 = color0 | (1 << v)
        new_color1 = color1
        if backtrack(new_unassigned, new_color0, new_color1):
            return True
            
        # Try color 1
        new_unassigned = unassigned & ~(1 << v)
        new_color0 = color0
        new_color1 = color1 | (1 << v)
        if backtrack(new_unassigned, new_color0, new_color1):
            return True
            
        return False

    return backtrack((1 << n) - 1, 0, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Try n from 9 to 12
    # n=9: C(9,5)=126 edges. 
    # n=10: C(10,5)=252 edges.
    # n=11: C(11,5)=462 edges.
    
    for n in range(9, 13):
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        num_edges_total = len(all_edges)
        
        # We want to find a small non-2-colorable subset.
        # Strategy: Start with a large random subset, then prune.
        
        # How large to start? 
        # If we start too large, pruning takes time.
        # If we start too small, it might be colorable.
        # Let's try starting with ~100 edges or 50% of all edges.
        
        restarts = 50
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            # Initialize with a random set of edges
            # Size: let's try 80 edges for n=9, 100 for n=10
            init_size = min(80, num_edges_total)
            if n > 9:
                init_size = min(120, num_edges_total)
                
            current_edges = random.sample(all_edges, init_size)
            
            # Check if it's non-2-colorable
            if is_2_colorable(n, current_edges):
                # If it is colorable, this start is useless for pruning to non-colorable.
                # We need to ADD edges to make it non-colorable? 
                # That's harder. 
                # Better: Start with a VERY dense graph. 
                # Let's restart this iteration with a denser graph if it was colorable.
                # Or just skip this restart?
                # Actually, if a random 80-edge graph on 9 vertices is colorable, 
                # we need more edges.
                # Let's force a denser start for n=9.
                if n == 9:
                    init_size = min(100, num_edges_total)
                    current_edges = random.sample(all_edges, init_size)
                    if is_2_colorable(n, current_edges):
                        continue # Skip this restart if still colorable
                
                if is_2_colorable(n, current_edges):
                    continue # Skip if still colorable

            # Now we have a non-2-colorable graph. Prune it.
            # Try to remove edges one by one.
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Shuffle to avoid bias
                random.shuffle(current_edges)
                for i in range(len(current_edges) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    e = current_edges[i]
                    # Try removing e
                    new_edges = current_edges[:i] + current_edges[i+1:]
                    if not is_2_colorable(n, new_edges):
                        # Removing e keeps it non-2-colorable
                        current_edges = new_edges
                        improved = True
                        break # Restart the loop with the new shorter list
            
            # Record result
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_n = n
                best_edges = [list(e) for e in current_edges]
                print(f"Found better: n={n}, edges={best_count}")
                
        # If we found a very small one, we might be able to stop early?
        # Theoretical lower bound is 29.
        if best_count <= 29:
            break

    if best_edges is None:
        # Fallback: Return a known valid one or a dense one
        # We must return a valid non-2-colorable hypergraph.
        # If the search failed, we need a guaranteed one.
        # The 51-edge construction is known. But we don't have it hardcoded.
        # Let's just return a dense random one on n=10 that we know is likely non-colorable.
        n = 10
        all_edges = list(combinations(range(n), 5))
        # Take first 100 edges
        edges = [list(e) for e in all_edges[:100]]
        # Verify it's non-colorable? 
        # If is_2_colorable(n, edges) is True, we have a problem.
        # But 100 edges on 10 vertices is very likely non-colorable.
        # Let's check.
        if is_2_colorable(n, edges):
             # Add more edges
             edges = [list(e) for e in all_edges]
        best_edges = edges
        best_n = n
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}

