import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety
    
    best_edges = None
    best_count = float('inf')
    best_n = None
    
    # Try n=11 first (more edges, potentially easier to find sparse non-colorable subgraph)
    # Then n=10, then n=9
    for n in [11, 10, 9]:
        if time.time() > deadline:
            break
            
        # 1. Generate all edges
        all_edges = [list(e) for e in combinations(range(n), 5)]
        num_edges = len(all_edges)
        
        # 2. Precompute bad_mask for all colorings
        # bad_mask[c] is a bitmask of edges that are monochromatic under coloring c
        bad_masks = [0] * (1 << n)
        
        # Precompute vertex masks for edges to speed up loop
        edge_masks = []
        for e in all_edges:
            mask_e = 0
            for v in e:
                mask_e |= (1 << v)
            edge_masks.append(mask_e)
            
        # For each coloring c
        for c in range(1 << n):
            bm = 0
            # For each edge
            for i in range(num_edges):
                mask_e = edge_masks[i]
                # Intersection of c and mask_e
                inter = c & mask_e
                # If inter == 0, all 0.
                # If inter == mask_e, all 1.
                if inter == 0 or inter == mask_e:
                    bm |= (1 << i)
            bad_masks[c] = bm
            
        # Quick check: if any coloring has no monochromatic edges, the full graph is colorable
        # In that case, no subset can be non-colorable? 
        # Wait, if the full graph is colorable, a subset is also colorable.
        # So if any bad_masks[c] == 0, we can skip this n entirely.
        if any(bm == 0 for bm in bad_masks):
            continue

        # 3. Define fast checker
        def is_non_colorable(active_mask):
            # active_mask is an int bitmask of edges present
            # If there is a coloring c such that (bad_masks[c] & active_mask) == 0, it is colorable.
            # So if no such c exists, it is non-colorable.
            for bm in bad_masks:
                if (bm & active_mask) == 0:
                    return False
            return True
            
        # 4. Greedy Reduction
        # Start with all edges
        current_mask = (1 << num_edges) - 1
        
        # Verify start is non-colorable
        if not is_non_colorable(current_mask):
            continue # Should not happen if we passed the quick check
            
        # Try multiple random reduction passes
        best_local_mask = current_mask
        best_local_count = num_edges
        
        # We can run this loop many times because checker is fast
        # Let's say 1000 restarts or until time
        restarts = 0
        while time.time() < deadline - 2:
            restarts += 1
            curr = current_mask
            changed = True
            while changed:
                changed = False
                # Shuffle edge indices
                indices = list(range(num_edges))
                random.shuffle(indices)
                for i in indices:
                    # Try removing edge i
                    if curr & (1 << i):
                        new_mask = curr & ~(1 << i)
                        if is_non_colorable(new_mask):
                            curr = new_mask
                            changed = True
                            # Don't break, continue trying to remove more
                            
            # Update best
            count = bin(curr).count('1')
            if count < best_local_count:
                best_local_count = count
                best_local_mask = curr
                
            # Optional: If we reach a very low count, we can stop early?
            # 51 is the target (best known).
            if best_local_count <= 51:
                break
                
        # Extract edges
        if best_local_count < best_count:
            best_count = best_local_count
            best_edges = [all_edges[i] for i in range(num_edges) if best_local_mask & (1 << i)]
            best_n = n
            
    if best_edges is None:
        # Fallback: return the full graph for n=9 which is known to be non-2-colorable
        # (Ramsey number R(3,3,3,3,3) > 9? Actually R(3,3)=6, R(3,3,3)=17.
        # Wait, 2-coloring means no monochromatic edge.
        # If n=9, is it non-2-colorable?
        # The 2-color Ramsey number R(3,3,3,3,3) is the smallest n such that any 2-coloring of K_n^{(5)} has a monochromatic K_3^{(5)}? No.
        # Property B: A hypergraph is 2-colorable if there is a 2-coloring with no monochromatic edge.
        # The question is: what is the smallest number of edges in a 5-uniform hypergraph that is NOT 2-colorable?
        # This is related to the property B number m(5).
        # It is known that m(5) is between 29 and 51.
        # The full graph K_9^{(5)} has 126 edges. Is it non-2-colorable?
        # If n=9, can we 2-color the vertices such that no 5-set is monochromatic?
        # If we color 4 vertices red and 5 blue, any 5-set that is all red is impossible (only 4 red).
        # Any 5-set that is all blue is possible if we pick the 5 blue vertices.
        # So if we have 5 blue vertices, the edge consisting of them is monochromatic blue.
        # So we must avoid having 5 vertices of the same color.
        # With 9 vertices, by Pigeonhole, one color has at least ceil(9/2)=5 vertices.
        # So there is always a monochromatic 5-set in K_9^{(5)}.
        # Thus K_9^{(5)} is NOT 2-colorable.
        # So the fallback is valid.
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
        
    return {'n': best_n, 'edges': best_edges}
