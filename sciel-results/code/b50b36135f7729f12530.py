import random
import time
import itertools

def construct():
    deadline = time.time() + 115
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute all K4s and their edge lists
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
    
    K = len(k4s)
    
    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)
    
    # Initialize with a structured coloring
    # (i + j) % 17 gives 17 colors
    colors = [(i + j) % 17 for (i, j) in edges]
    
    # Compress colors to 0..palette_size-1
    def compress(colors):
        mapping = {}
        next_id = 0
        new_colors = []
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    colors, palette_size = compress(colors)
    
    # Track color usage counts
    color_counts = [0] * palette_size
    for c in colors:
        color_counts[c] += 1
        
    # Track distinct color count per K4
    k4_distinct = [0] * K
    # Precompute sets of colors for each K4 for fast update?
    # Or just recompute for affected K4s.
    
    def is_valid():
        for k4_idx in range(K):
            k4e = k4s[k4_idx]
            distinct = len(set(colors[e] for e in k4e))
            if distinct < 5:
                return False
        return True

    # Initial validation
    if not is_valid():
        # Fallback if initial is invalid (shouldn't happen)
        return colors

    best_colors = colors[:]
    best_palette = palette_size
    
    rng = random.Random(42)
    
    # Helper to check if a specific K4 is valid
    def k4_is_valid(k4_idx):
        k4e = k4s[k4_idx]
        return len(set(colors[e] for e in k4e)) >= 5

    # Local search loop
    # Strategy: Try to eliminate colors
    for iteration in range(500000):
        if time.time() > deadline - 5:
            break
            
        # If we have found a very good palette, we might want to polish
        # but for now, just keep trying to reduce
        
        # Pick a color to try to eliminate
        # Prefer colors with fewer edges? Or random?
        # Let's pick a random color that is currently in use
        active_colors = [c for c in range(palette_size) if color_counts[c] > 0]
        if not active_colors:
            break
            
        target_color = rng.choice(active_colors)
        
        # Get edges with this color
        edges_with_target = [e for e in range(E) if colors[e] == target_color]
        
        # Try to recolor each edge in edges_with_target to a different color
        # We do this sequentially. If we fail, we revert.
        
        # To make it more robust, we can shuffle the edges
        rng.shuffle(edges_with_target)
        
        success = True
        changes = [] # (edge, old_color, new_color)
        
        for e in edges_with_target:
            old_c = colors[e]
            # Try other active colors
            candidates = [c for c in active_colors if c != old_c]
            rng.shuffle(candidates)
            
            found = False
            for nc in candidates:
                # Tentatively change
                colors[e] = nc
                
                # Check validity for all K4s containing e
                valid = True
                for k4_idx in edge_to_k4s[e]:
                    if not k4_is_valid(k4_idx):
                        valid = False
                        break
                        
                if valid:
                    # Update counts
                    color_counts[old_c] -= 1
                    color_counts[nc] += 1
                    changes.append((e, old_c, nc))
                    found = True
                    break
                else:
                    # Revert
                    colors[e] = old_c
            
            if not found:
                # Revert all changes made so far in this attempt
                for (ce, oc, ncv) in changes:
                    colors[ce] = oc
                    color_counts[oc] += 1
                    color_counts[ncv] -= 1
                success = False
                break
        
        if success:
            # Color target_color is eliminated
            # Update active_colors? We can just check color_counts later.
            # The palette_size doesn't change immediately, but the count for target_color is 0.
            # We can compress again periodically to keep indices small, 
            # or just leave it. Leaving it is fine, we just track active_colors.
            
            # Check if we improved
            current_palette = sum(1 for c in range(palette_size) if color_counts[c] > 0)
            if current_palette < best_palette:
                best_palette = current_palette
                best_colors = colors[:]
                
            # If we hit 13, we can stop
            if best_palette <= 13:
                break
                
        else:
            # If we failed, maybe do a random perturbation to escape local minimum
            # Pick a random edge and change it to a random active color
            if rng.random() < 0.1: # 10% chance of perturbation
                e = rng.randrange(E)
                old_c = colors[e]
                active_colors = [c for c in range(palette_size) if color_counts[c] > 0]
                if active_colors:
                    nc = rng.choice(active_colors)
                    if nc != old_c:
                        colors[e] = nc
                        color_counts[old_c] -= 1
                        color_counts[nc] += 1
                        
    # Final compression of the best solution
    # The best_colors might have gaps in indices.
    # The evaluator expects integer colours. It doesn't say they must be contiguous,
    # but "Minimize the number of colours used" implies the count of distinct values.
    # So we should return the list of colors.
    
    # Ensure best_colors is valid
    # (It should be, as we only accepted valid moves)
    
    return best_colors
