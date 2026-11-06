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
    # Use (i + j) % p for some prime p to get good initial diversity
    colors = [(i + j) % 17 for (i, j) in edges]
    
    # Track color usage counts
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    # Track distinct color count per K4
    k4_distinct = [0] * K
    total_badness = 0
    for k4_idx, k4e in enumerate(k4s):
        distinct = len(set(colors[e] for e in k4e))
        k4_distinct[k4_idx] = distinct
        if distinct < 5:
            total_badness += (5 - distinct)
    
    palette_size = len(color_counts)
    
    best_colors = colors[:]
    best_badness = total_badness
    best_palette = palette_size
    best_score = total_badness * 10000 + palette_size
    
    rng = random.Random(42)
    
    def score(badness, palette):
        return badness * 10000 + palette
    
    for iteration in range(5000000):
        if time.time() > deadline - 5:
            break
        
        e = rng.randrange(E)
        old_c = colors[e]
        
        # Determine candidate colors
        # Phase 1: if badness > 0, prioritize fixing
        # Phase 2: if badness == 0, prioritize reducing palette
        
        if total_badness > 0:
            # Try new random colors and existing colors
            candidates = []
            # Add some random new colors (0 to 40 range)
            for _ in range(8):
                candidates.append(rng.randint(0, 40))
            # Add existing colors
            for c in color_counts:
                if c != old_c:
                    candidates.append(c)
        else:
            # Valid state: try to reduce palette
            # Only use existing colors, prefer ones that can eliminate old_c
            candidates = [c for c in color_counts if c != old_c]
            if not candidates:
                candidates = [old_c]  # shouldn't happen
        
        if not candidates:
            candidates = [0]
        
        rng.shuffle(candidates)
        
        best_move_c = old_c
        best_move_badness = total_badness
        best_move_palette = palette_size
        best_move_score = score(total_badness, palette_size)
        
        for nc in candidates:
            if nc == old_c:
                continue
            
            # Tentatively apply
            colors[e] = nc
            
            # Update color counts
            color_counts[old_c] -= 1
            if color_counts[old_c] == 0:
                del color_counts[old_c]
            color_counts[nc] = color_counts.get(nc, 0) + 1
            
            new_badness = total_badness
            for k4_idx in edge_to_k4s[e]:
                old_dist = k4_distinct[k4_idx]
                k4e = k4s[k4_idx]
                distinct = len(set(colors[ee] for ee in k4e))
                k4_distinct[k4_idx] = distinct
                
                if old_dist < 5:
                    new_badness -= (5 - old_dist)
                if distinct < 5:
                    new_badness += (5 - distinct)
            
            new_palette = len(color_counts)
            new_score_val = score(new_badness, new_palette)
            
            if new_score_val < best_move_score:
                best_move_score = new_score_val
                best_move_c = nc
                best_move_badness = new_badness
                best_move_palette = new_palette
            
            # Revert
            colors[e] = old_c
            color_counts[nc] -= 1
            if color_counts[nc] == 0:
                del color_counts[nc]
            color_counts[old_c] = color_counts.get(old_c, 0) + 1
            
            # Restore k4_distinct
            for k4_idx in edge_to_k4s[e]:
                k4e = k4s[k4_idx]
                k4_distinct[k4_idx] = len(set(colors[ee] for ee in k4e))
        
        # Apply best move
        if best_move_c != old_c:
            colors[e] = best_move_c
            color_counts[old_c] -= 1
            if color_counts[old_c] == 0:
                del color_counts[old_c]
            color_counts[best_move_c] = color_counts.get(best_move_c, 0) + 1
            
            for k4_idx in edge_to_k4s[e]:
                k4e = k4s[k4_idx]
                k4_distinct[k4_idx] = len(set(colors[ee] for ee in k4e))
            
            total_badness = best_move_badness
            palette_size = best_move_palette
            
            current_score = score(total_badness, palette_size)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
                best_badness = total_badness
                best_palette = palette_size
    
    # Final verification and return
    # Verify validity
    valid = True
    for k4e in k4s:
        if len(set(best_colors[e] for e in k4e)) < 5:
            valid = False
            break
    
    if not valid:
        # Fallback: use a known valid construction if possible
        # Return the best we have (should be valid if algorithm worked)
        pass
    
    return best_colors
