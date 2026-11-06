"""Edge-colourings of K16 in which every K4 sees five colours."""
import time
import itertools
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Return 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before the 150s limit
    
    n = 16
    
    # Precompute edges and K4s
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Map (i, j) to edge index
    ij_to_edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute for each K4, the indices of its 6 edges
    k4s = list(itertools.combinations(range(n), 4))
    k4_edge_indices = []
    for verts in k4s:
        edges_in_k4 = []
        for i in range(4):
            for j in range(i + 1, 4):
                u = verts[i]
                v = verts[j]
                edges_in_k4.append(ij_to_edge_idx[(u, v)])
        k4_edge_indices.append(edges_in_k4)
        
    # Map edge index to list of K4 indices containing that edge
    # This helps in incremental checks
    edge_to_k4s = {idx: [] for idx in range(len(edges))}
    for k4_idx, e_idxs in enumerate(k4_edge_indices):
        for e_idx in e_idxs:
            edge_to_k4s[e_idx].append(k4_idx)

    def is_valid_full(colors):
        """Check if every K4 has at least 5 distinct colors."""
        for e_idxs in k4_edge_indices:
            c = [colors[e_idxs[0]], colors[e_idxs[1]], colors[e_idxs[2]], 
                 colors[e_idxs[3]], colors[e_idxs[4]], colors[e_idxs[5]]]
            if len(set(c)) < 5:
                return False
        return True

    def is_valid_incremental(colors, changed_edges):
        """
        Check validity only for K4s that contain at least one changed edge.
        changed_edges: set of edge indices that were modified.
        """
        # Collect all K4s to check
        k4s_to_check = set()
        for e_idx in changed_edges:
            k4s_to_check.update(edge_to_k4s[e_idx])
        
        for k4_idx in k4s_to_check:
            e_idxs = k4_edge_indices[k4_idx]
            c = [colors[e_idxs[0]], colors[e_idxs[1]], colors[e_idxs[2]], 
                 colors[e_idxs[3]], colors[e_idxs[4]], colors[e_idxs[5]]]
            if len(set(c)) < 5:
                return False
        return True

    # Initial coloring: (i + j) % 17
    colors = [0] * len(edges)
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
        
    # Verify initial
    if not is_valid_full(colors):
        # Fallback: return the additive coloring as is (should be valid)
        return colors

    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    # Merging loop
    # We try to merge colors to reduce the count.
    # We restart the merge process if stuck, trying different random orders of color pairs.
    
    rng = random.Random(42)
    
    for attempt in range(100): # Limit attempts to avoid infinite loop if stuck
        if time.time() > deadline:
            break
            
        current_colors = best_colors[:]
        current_used = set(current_colors)
        
        # Try to merge
        progress = True
        while progress:
            if time.time() > deadline:
                return best_colors
            
            progress = False
            # Shuffle the order of colors to try to escape local minima
            colors_list = list(current_used)
            rng.shuffle(colors_list)
            
            for c_from in colors_list:
                if time.time() > deadline:
                    return best_colors
                if c_from not in current_used:
                    continue
                
                # Find edges with c_from
                edges_with_from = [idx for idx, c in enumerate(current_colors) if c == c_from]
                
                for c_to in colors_list:
                    if c_from == c_to:
                        continue
                    if c_to not in current_used:
                        continue
                    
                    # Create temporary changes
                    # We only need to check K4s affected by edges changing from c_from to c_to
                    # But we also need to ensure we don't break K4s that already had < 5?
                    # No, current state is valid. So we only need to check if the change creates a violation.
                    
                    # Identify changed edges
                    changed_edges = set(edges_with_from)
                    
                    # Apply change to a copy
                    temp_colors = current_colors[:]
                    for idx in edges_with_from:
                        temp_colors[idx] = c_to
                    
                    # Check validity incrementally
                    if is_valid_incremental(temp_colors, changed_edges):
                        # Accept merge
                        current_colors = temp_colors
                        current_used.remove(c_from)
                        progress = True
                        # Update best if better
                        if len(current_used) < best_num_colors:
                            best_colors = current_colors[:]
                            best_num_colors = len(current_used)
                        break # Break inner loop, restart outer with new state
                
                if progress:
                    break # Break outer loop, restart with new state
                    
        # If no progress in this attempt, try a random perturbation?
        # Or just break and try another random seed/order in the next attempt.
        # For now, we just rely on the shuffle in the next attempt.
        
    # Final check
    if not is_valid_full(best_colors):
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return best_colors

# EVOLVE-BLOCK-END
