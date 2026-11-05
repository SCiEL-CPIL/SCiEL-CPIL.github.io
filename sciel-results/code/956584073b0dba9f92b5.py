import random
import time
import numpy as np

def construct():
    # Target n. The known record is 50. We aim for 50.
    n = 50
    num_colors = 4
    
    # Precompute edge indices and triangle neighbors
    # Edge index for (i, j) with i < j:
    # idx = i*n - i*(i+1)//2 + j - i - 1
    # Or simply build a map.
    edge_to_ij = []
    idx_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(edge_to_ij)
            edge_to_ij.append((i, j))
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
    
    num_edges = len(edge_to_ij)
    
    # Precompute for each edge e, the list of (k, idx_ik, idx_jk)
    # such that (i, j, k) is a triangle.
    # i, j are the vertices of edge e.
    tri_neighbors = [[] for _ in range(num_edges)]
    
    for e, (i, j) in enumerate(edge_to_ij):
        for k in range(n):
            if k == i or k == j:
                continue
            if i < k:
                idx_ik = idx_map[(i, k)]
            else:
                idx_ik = idx_map[(k, i)]
            if j < k:
                idx_jk = idx_map[(j, k)]
            else:
                idx_jk = idx_map[(k, j)]
            tri_neighbors[e].append((k, idx_ik, idx_jk))

    def count_mono_triangles(colors):
        """Count total monochromatic triangles."""
        count = 0
        for e in range(num_edges):
            c_e = colors[e]
            for k, idx_ik, idx_jk in tri_neighbors[e]:
                if colors[idx_ik] == c_e and colors[idx_jk] == c_e:
                    count += 1
        return count // 3

    def get_delta(colors, e, new_color):
        """
        Calculate change in mono triangle count if edge e is changed to new_color.
        Returns the delta.
        """
        old_color = colors[e]
        if old_color == new_color:
            return 0
        
        delta = 0
        for k, idx_ik, idx_jk in tri_neighbors[e]:
            c_ik = colors[idx_ik]
            c_jk = colors[idx_jk]
            
            # Was it mono?
            was_mono = (c_ik == old_color and c_jk == old_color)
            # Is it mono now?
            is_mono = (c_ik == new_color and c_jk == new_color)
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
                
        return delta

    # Local Search
    start_time = time.time()
    time_limit = 115 # Leave 5s buffer
    
    # Fallback 32-vertex solution
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    # Strategy:
    # Try to find a valid coloring for n=50.
    # Use a "First Fit" or "Greedy" local search with random restarts.
    # To speed up, we maintain the score.
    
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Calculate initial score
        score = count_mono_triangles(colors)
        
        if score == 0:
            return {'n': n, 'colors': colors}
            
        # Local search loop
        # We will try to reduce score to 0.
        # We pick a random edge, try all 4 colors, pick the best one.
        # If no improvement, we might be stuck.
        # To avoid getting stuck in a local minimum, we allow "neutral" moves (delta=0) 
        # or "worse" moves (delta>0) with a small probability (simulated annealing style), 
        # or just restart.
        
        # Given the structure of R(3,3,3,3), simple hill climbing often gets stuck.
        # We will use a strategy: 
        # 1. Find an edge involved in a mono triangle.
        # 2. Recolor it to the color that minimizes the number of mono triangles 
        #    involving that edge.
        # 3. If no edge can be improved, restart.
        
        max_iters = 20000
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            if score == 0:
                return {'n': n, 'colors': colors}
                
            # Pick a random edge
            e = random.randint(0, num_edges - 1)
            
            # Check if this edge is in a mono triangle
            # If not, it doesn't contribute to score directly, but changing it might affect neighbors.
            # However, if we only change edges that are in mono triangles, we guarantee progress 
            # if an improvement exists for that edge.
            
            # Let's check if edge e is in any mono triangle.
            # We can just calculate the delta for all 4 colors.
            # If the current color is part of a mono triangle, there is a high chance 
            # another color is better.
            
            current_color = colors[e]
            
            # Calculate delta for each of the 4 colors
            best_delta = 0 # We want to minimize score, so delta < 0 is good.
            # Actually, we want the *most negative* delta.
            # If all deltas are >= 0, we are stuck at this edge.
            
            # Optimization: Only consider changing if the current color is involved in a mono triangle?
            # No, changing a non-mono edge to a color that creates mono triangles is bad.
            # Changing a non-mono edge to a color that *removes* mono triangles is impossible 
            # because it wasn't creating them.
            # So, only edges in mono triangles can potentially reduce the score by changing.
            # Wait, if edge e is NOT in a mono triangle, changing it to a new color c:
            #   - It might create new mono triangles (delta > 0).
            #   - It cannot remove existing mono triangles (delta cannot be < 0).
            # So, we should ONLY consider changing edges that are currently in at least one mono triangle.
            
            # Check if e is in a mono triangle
            in_mono = False
            for k, idx_ik, idx_jk in tri_neighbors[e]:
                if colors[idx_ik] == current_color and colors[idx_jk] == current_color:
                    in_mono = True
                    break
            
            if not in_mono:
                continue # Skip this edge, it's "good" as is.
                
            # Find the best new color
            best_new_color = -1
            best_delta_val = 0 # We want delta < 0
            
            for c in range(4):
                if c == current_color:
                    continue
                d = get_delta(colors, e, c)
                if d < best_delta_val:
                    best_delta_val = d
                    best_new_color = c
                    
            if best_new_color != -1 and best_delta_val < 0:
                # Apply move
                colors[e] = best_new_color
                score += best_delta_val
            else:
                # Stuck? 
                # If we scan many edges and find no improvement, we are in a local minimum.
                # We can try a random jump (kick) or restart.
                # To keep it simple and fast, we will just restart if we've tried 
                # many iterations without success? 
                # Actually, the loop above just skips non-mono edges.
                # If we find a mono edge but no improvement, we skip it.
                # If we cycle through all edges and find no improvement, we restart.
                # To detect "no improvement", we can count how many times we've skipped 
                # or tried and failed.
                # Simpler: Just run for a fixed number of iterations. 
                # If score doesn't drop to 0, restart.
                # But we need to ensure we don't just loop forever on a bad config.
                # The `max_iters` limit handles this.
                pass

    # If we failed to find 50 in time, fall back to 32
    return {'n': 32, 'colors': colors_32}
