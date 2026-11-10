import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return
    
    n = 16
    # Precompute all K4s and their edges
    # Vertices 0..15
    vertices = list(range(n))
    k4s = list(itertools.combinations(vertices, 4))
    
    # Map edge (i, j) with i < j to index 0..119
    def edge_idx(i, j):
        if i > j: i, j = j, i
        # i < j
        # Number of edges before i is i*(n-1) - i*(i-1)/2 ... no simpler:
        # Sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # Let's just use a dict or a precomputed matrix for speed
        pass

    # Precompute edge indices for all pairs
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_map[(i, j)] = idx
            idx += 1
            
    # For each K4, store the list of edge indices
    k4_edges = []
    for combo in k4s:
        edges = []
        for u, v in itertools.combinations(combo, 2):
            edges.append(edge_map[(u, v)])
        k4_edges.append(edges)
        
    # For each edge, store the list of K4 indices that contain it
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    # Function to count distinct colors in a K4
    def count_distinct_colors(k4_idx, colors):
        edges = k4_edges[k4_idx]
        cset = set(colors[e] for e in edges)
        return len(cset)

    # Function to evaluate total violations
    def total_violations(colors):
        viol = 0
        for k4_idx in range(len(k4s)):
            if count_distinct_colors(k4_idx, colors) < 5:
                viol += 1
        return viol

    best_colors = None
    best_viol = float('inf')
    best_num_colors = 17 # Fallback to seed-like behavior if we fail completely? 
                         # But we want to minimize. Let's start with a random 14-coloring.
    
    # Try to find a solution with K colors
    for K in [14, 13]:
        # Initialize random coloring with K colors
        colors = [random.randint(0, K-1) for _ in range(120)]
        
        # Precompute violations per K4 to speed up updates
        # viol_count[k4_idx] = number of distinct colors - 5 if < 5 else 0? 
        # Actually, we just need to know if it's valid.
        # Let's maintain an array `k4_valid` of booleans or violation counts.
        # To make it faster, let's maintain `distinct_counts` for each K4.
        # distinct_counts[k4_idx] = number of distinct colors in that K4.
        
        distinct_counts = [0] * len(k4s)
        for k4_idx in range(len(k4s)):
            distinct_counts[k4_idx] = count_distinct_colors(k4_idx, colors)
            
        total_viol = sum(1 for c in distinct_counts if c < 5)
        
        if total_viol == 0:
            best_colors = colors
            best_num_colors = K
            break # Found a valid coloring with K colors, try smaller K? Or stop if K is small enough.
            # Actually, if K=14 works, we should try K=13 next. The loop does that.
            # But if K=14 works, we can stop searching for K=14 and move to 13.
            # The loop structure `for K in [14, 13]` handles this.
            # Wait, if K=14 works, we set best. Then loop continues to K=13.
            # If K=13 works, we update best.
            
        # Local Search
        current_viol = total_viol
        current_colors = colors[:]
        
        # Simple hill climbing with random restarts if stuck
        # We will run local search for a fixed amount of time or steps
        steps = 0
        max_steps = 20000
        no_improve = 0
        max_no_improve = 5000
        
        while steps < max_steps and no_improve < max_no_improve:
            if time.time() > deadline:
                break
                
            # Pick a random K4 that is currently invalid
            invalid_k4s = [i for i in range(len(k4s)) if distinct_counts[i] < 5]
            if not invalid_k4s:
                break # Found a valid coloring!
                
            k4_idx = random.choice(invalid_k4s)
            edges = k4_edges[k4_idx]
            
            # Pick a random edge in this K4 to change
            e_idx = random.choice(edges)
            
            old_color = current_colors[e_idx]
            # Choose a new color
            # Strategy: pick a color that appears less often in this K4? 
            # Or just random from 0..K-1
            new_color = random.randint(0, K-1)
            if new_color == old_color:
                continue
                
            # Update distinct_counts for all K4s containing e_idx
            # We need to recalculate distinct colors for these K4s
            # This is expensive if done naively. 
            # Optimization: 
            # For each affected K4, we can maintain a count of color frequencies?
            # That's complex. 
            # Given 91 K4s per edge, and 120 edges, total K4s 1820.
            # Recalculating distinct colors for 91 K4s involves checking 6 edges each.
            # 91 * 6 = 546 operations. Very fast.
            
            affected_k4s = edge_to_k4s[e_idx]
            
            # Calculate new distinct counts for affected K4s
            # We need the colors of the other 5 edges in each K4
            # Let's just recalculate from scratch for the affected K4s.
            # Store old distinct counts to check for violation change
            old_distincts = {k4: distinct_counts[k4] for k4 in affected_k4s}
            
            current_colors[e_idx] = new_color
            
            new_total_viol_delta = 0
            for k4 in affected_k4s:
                new_dist = count_distinct_colors(k4, current_colors)
                old_dist = distinct_counts[k4]
                
                # Update violation status
                old_was_invalid = old_dist < 5
                new_is_invalid = new_dist < 5
                
                if old_was_invalid and not new_is_invalid:
                    new_total_viol_delta -= 1
                elif not old_was_invalid and new_is_invalid:
                    new_total_viol_delta += 1
                    
                distinct_counts[k4] = new_dist
                
            new_total_viol = current_viol + new_total_viol_delta
            
            if new_total_viol < current_viol:
                current_viol = new_total_viol
                no_improve = 0
                steps += 1
            else:
                # Revert
                current_colors[e_idx] = old_color
                for k4 in affected_k4s:
                    distinct_counts[k4] = old_distincts[k4]
                no_improve += 1
                steps += 1
                
        if current_viol == 0:
            best_colors = current_colors
            best_num_colors = K
            # If we found a valid coloring with K, we should try smaller K if possible.
            # The loop will try the next smaller K.
            # But we should probably break out of the local search and let the outer loop handle it.
            # Actually, if K=14 works, we want to see if 13 works.
            # So we don't break the outer loop immediately, we just update best and continue.
            # However, finding a valid 14-coloring is a good result.
            # If we find a valid 13-coloring, that's better.
            
    # If no valid coloring found with 14 or 13, we might have a valid one with 15 or 16?
    # The seed uses 17.
    # Let's try a quick search for 15 colors if we failed 14/13.
    if best_colors is None:
        for K in [15, 16]:
            colors = [random.randint(0, K-1) for _ in range(120)]
            distinct_counts = [count_distinct_colors(i, colors) for i in range(len(k4s))]
            current_viol = sum(1 for c in distinct_counts if c < 5)
            
            if current_viol == 0:
                best_colors = colors
                best_num_colors = K
                break
                
            # Short local search
            steps = 0
            max_steps = 5000
            while steps < max_steps:
                if time.time() > deadline:
                    break
                invalid_k4s = [i for i in range(len(k4s)) if distinct_counts[i] < 5]
                if not invalid_k4s:
                    break
                k4_idx = random.choice(invalid_k4s)
                edges = k4_edges[k4_idx]
                e_idx = random.choice(edges)
                old_color = colors[e_idx]
                new_color = random.randint(0, K-1)
                if new_color == old_color: continue
                
                affected_k4s = edge_to_k4s[e_idx]
                old_distincts = {k4: distinct_counts[k4] for k4 in affected_k4s}
                
                colors[e_idx] = new_color
                delta = 0
                for k4 in affected_k4s:
                    new_dist = count_distinct_colors(k4, colors)
                    old_dist = distinct_counts[k4]
                    old_inv = old_dist < 5
                    new_inv = new_dist < 5
                    if old_inv and not new_inv: delta -= 1
                    elif not old_inv and new_inv: delta += 1
                    distinct_counts[k4] = new_dist
                    
                new_viol = current_viol + delta
                if new_viol < current_viol:
                    current_viol = new_viol
                    steps += 1
                else:
                    colors[e_idx] = old_color
                    for k4 in affected_k4s:
                        distinct_counts[k4] = old_distincts[k4]
                    steps += 1
                    
            if current_viol == 0:
                best_colors = colors
                best_num_colors = K
                break

    # Fallback: Use the seed coloring (17 colors) if nothing better found
    if best_colors is None:
        best_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        best_num_colors = 17
        
    # Verify best_colors just in case
    # (Optional, but good for safety)
    # return best_colors
    return best_colors
