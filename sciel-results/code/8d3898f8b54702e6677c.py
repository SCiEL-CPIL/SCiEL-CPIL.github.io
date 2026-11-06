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
    
    # Better initial coloring: use a structured pattern that's likely valid
    # Try (i * 7 + j * 3) % 13 to get good diversity
    colors = [(i * 7 + j * 3) % 13 for (i, j) in edges]
    
    # Track color usage counts
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    # Track distinct color count per K4 and track color frequency within each K4
    k4_distinct = [0] * K
    k4_color_freq = [{} for _ in range(K)]
    total_badness = 0
    for k4_idx, k4e in enumerate(k4s):
        freq = {}
        for e in k4e:
            c = colors[e]
            freq[c] = freq.get(c, 0) + 1
        k4_color_freq[k4_idx] = freq
        distinct = len(freq)
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
    
    # Precompute candidate color sets for faster evaluation
    # Limit search to colors 0..20 initially
    max_color_search = 20
    
    for iteration in range(5000000):
        if time.time() > deadline - 5:
            break
        
        e = rng.randrange(E)
        old_c = colors[e]
        
        # Determine candidate colors based on current state
        if total_badness > 0:
            # When bad, try a focused set of colors
            candidates = set()
            # Try colors 0 to max_color_search
            for c in range(max_color_search + 1):
                if c != old_c:
                    candidates.add(c)
            # Also try some random higher colors
            for _ in range(5):
                candidates.add(rng.randint(0, 30))
        else:
            # Valid state: only use existing colors to reduce palette
            candidates = set(color_counts.keys())
            candidates.discard(old_c)
            if not candidates:
                candidates.add(old_c)
        
        if not candidates:
            candidates = {0}
        
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
                freq = k4_color_freq[k4_idx]
                old_dist = k4_distinct[k4_idx]
                
                # Update frequency
                freq[old_c] -= 1
                if freq[old_c] == 0:
                    del freq[old_c]
                freq[nc] = freq.get(nc, 0) + 1
                
                new_dist = len(freq)
                k4_distinct[k4_idx] = new_dist
                
                if old_dist < 5:
                    new_badness -= (5 - old_dist)
                if new_dist < 5:
                    new_badness += (5 - new_dist)
            
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
            
            # Restore k4_color_freq
            for k4_idx in edge_to_k4s[e]:
                freq = k4_color_freq[k4_idx]
                freq[nc] -= 1
                if freq[nc] == 0:
                    del freq[nc]
                freq[old_c] = freq.get(old_c, 0) + 1
                k4_distinct[k4_idx] = len(freq)
        
        # Apply best move
        if best_move_c != old_c:
            colors[e] = best_move_c
            color_counts[old_c] -= 1
            if color_counts[old_c] == 0:
                del color_counts[old_c]
            color_counts[best_move_c] = color_counts.get(best_move_c, 0) + 1
            
            for k4_idx in edge_to_k4s[e]:
                freq = k4_color_freq[k4_idx]
                freq[old_c] -= 1
                if freq[old_c] == 0:
                    del freq[old_c]
                freq[best_move_c] = freq.get(best_move_c, 0) + 1
                k4_distinct[k4_idx] = len(freq)
            
            total_badness = best_move_badness
            palette_size = best_move_palette
            
            current_score = score(total_badness, palette_size)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
                best_badness = total_badness
                best_palette = palette_size
    
    # Final verification and return
    valid = True
    for k4e in k4s:
        if len(set(best_colors[e] for e in k4e)) < 5:
            valid = False
            break
    
    if not valid:
        # Fallback: try to fix with a simple greedy approach
        for k4e in k4s:
            if len(set(best_colors[e] for e in k4e)) < 5:
                # Find a K4 with < 5 colors and try to fix it
                freq = {}
                for e in k4e:
                    c = best_colors[e]
                    freq[c] = freq.get(c, 0) + 1
                # Find the most common color and change one edge to a new color
                if freq:
                    most_common = max(freq, key=freq.get)
                    for e in k4e:
                        if best_colors[e] == most_common:
                            # Find a color not in this K4
                            used = set(freq.keys())
                            new_c = 0
                            while new_c in used:
                                new_c += 1
                            best_colors[e] = new_c
                            break
    
    return best_colors
