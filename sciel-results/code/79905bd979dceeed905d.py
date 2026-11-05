import time
import random
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)

    def is_valid_assignment(colors, edge_idx, color):
        """
        Check if assigning 'color' to 'edge_idx' results in a valid state 
        for all K4s containing that edge, assuming other edges have 
        colors in 'colors' (or -1).
        """
        for k4_idx in edge_to_k4s[edge_idx]:
            e_indices = k4_edges[k4_idx]
            # Get colors of the 6 edges in this K4
            # One of them is edge_idx, which we are setting to 'color'
            # Others are from 'colors' array
            k4_colors = []
            for e in e_indices:
                if e == edge_idx:
                    k4_colors.append(color)
                else:
                    k4_colors.append(colors[e])
            
            # Check if all are colored (no -1)
            if -1 in k4_colors:
                # We only check validity if the K4 is fully colored?
                # Actually, if a K4 is not fully colored, it might become invalid later.
                # But for the greedy step, we just ensure that if it IS fully colored, it's valid.
                # If it's not fully colored, we can't know yet.
                # However, if we assign a color that *prevents* future validity, that's hard to check.
                # Standard greedy: ensure that if the K4 is complete, it's valid.
                # If the K4 is incomplete, we assume we can fix it later?
                # No, if we have 5 edges colored with 4 distinct colors, and the 6th is uncolored,
                # the 6th MUST be a new color. If the palette is exhausted or constrained, it might fail.
                # But for now, just check if the current partial assignment is "compatible" with validity.
                # Actually, if -1 is present, the distinct count is unknown.
                # Let's just return True if -1 is present, unless the colored part already violates?
                # A partial assignment can never "violate" in the sense of final distinct count < 5
                # unless the colored edges already force it?
                # Example: 5 edges colored. Colors: 1,1,1,2,3. Distinct so far: 3.
                # If we add 4, distinct becomes 4. Invalid.
                # So we should check the potential distinct count.
                
                # Let's collect non-(-1) colors
                non_neg = [c for c in k4_colors if c != -1]
                if len(non_neg) < 5:
                    # Can't determine final count yet, but check if it's already impossible?
                    # If we have 5 edges, and they have <= 4 distinct colors, the 6th must add a new one.
                    # If we have 4 edges, they can have <= 4 distinct.
                    # It's hard to prune early without lookahead.
                    # Let's just check if the set of ALL 6 (assuming -1 is "wildcard") 
                    # can possibly reach 5 distinct?
                    # Actually, simpler: just check if the current set of colored edges
                    # has a frequency distribution that makes 5 distinct impossible?
                    # No, let's just check the final condition if all are colored.
                    # If not all colored, return True (optimistic).
                    # This might lead to dead ends, but random restarts handle that.
                    return True
            
            # All 6 are colored
            distinct_count = len(set(k4_colors))
            if distinct_count < 5:
                return False
        
        return True

    best_result = None
    best_colors_used = 100  # Start high
    
    # Baseline fallback
    baseline = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    # Verify baseline is valid (it is, per prompt)
    
    for target_C in [13, 14, 15, 16]:
        if time.time() > deadline - 10:
            break
            
        # Try to find a coloring with exactly target_C colors (or fewer, but we use 0..target_C-1)
        attempts = 0
        max_attempts_per_C = 2000 if target_C == 13 else 5000
        
        while attempts < max_attempts_per_C:
            if time.time() > deadline - 5:
                break
                
            attempts += 1
            random.shuffle(edges) # Randomize edge order for this attempt
            colors = [-1] * num_edges
            success = True
            
            # Map original edge index to position in shuffled list?
            # It's easier to just iterate through a shuffled list of edge indices.
            edge_order = list(range(num_edges))
            random.shuffle(edge_order)
            
            for e_idx in edge_order:
                # Try random colors from 0 to target_C - 1
                # Shuffle colors too?
                color_options = list(range(target_C))
                random.shuffle(color_options)
                
                assigned = False
                for c in color_options:
                    if is_valid_assignment(colors, e_idx, c):
                        colors[e_idx] = c
                        assigned = True
                        break
                
                if not assigned:
                    success = False
                    break
            
            if success:
                # Verify the full coloring
                valid = True
                for k4_idx in range(len(k4s)):
                    e_indices = k4_edges[k4_idx]
                    k4_colors = [colors[e] for e in e_indices]
                    if len(set(k4_colors)) < 5:
                        valid = False
                        break
                
                if valid:
                    # Count distinct colors used
                    used = set(colors)
                    num_used = len(used)
                    if num_used < best_colors_used:
                        best_colors_used = num_used
                        best_result = colors[:]
                    
                    # If we found a coloring with 13 colors, that's likely optimal or near it.
                    # We can return immediately if we are confident, or continue to see if we can get 12?
                    # Bounds say 13 <= f(16). So 13 is the minimum possible.
                    if num_used <= 13:
                        return best_result
                    # If we found 14, we might still try to find 13.
                    # But 13 is the lower bound. If we found 13, return.
                    # If we found 14, continue trying for 13?
                    # The loop structure tries target_C=13 first.
                    # So if we are in target_C=13 and success, num_used <= 13.
                    # So return.
                    
        # If target_C=13 failed, loop goes to 14.
        # If we have a valid result with 14 colors, we hold it.
        # If we later find 13, we update.
        
    if best_result is not None:
        return best_result
    else:
        return baseline
