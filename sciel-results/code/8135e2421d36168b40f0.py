import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    # Pairs (i, j) in lexicographic order
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # Map pair to index
    pair_index = {p: idx for idx, p in enumerate(pairs)}
    
    # Initial coloring: (i + j) % 17
    colors = [(i + j) % 17 for i, j in pairs]
    num_colors = len(set(colors))
    
    # Precompute all K4s
    k4s = list(combinations(range(n), 4))
    
    def get_k4_colors(k4, col_list):
        # k4 is a tuple of 4 vertices
        v1, v2, v3, v4 = k4
        c1 = col_list[pair_index[(v1, v2)]]
        c2 = col_list[pair_index[(v1, v3)]]
        c3 = col_list[pair_index[(v1, v4)]]
        c4 = col_list[pair_index[(v2, v3)]]
        c5 = col_list[pair_index[(v2, v4)]]
        c6 = col_list[pair_index[(v3, v4)]]
        return {c1, c2, c3, c4, c5, c6}

    def is_valid_merge(c1, c2, col_list):
        # Check if merging c1 and c2 is valid
        # A K4 is invalid if it has exactly 5 distinct colors, and both c1 and c2 are among them.
        for k4 in k4s:
            k4_colors = get_k4_colors(k4, col_list)
            if len(k4_colors) == 5:
                if c1 in k4_colors and c2 in k4_colors:
                    return False
        return True

    def try_merge():
        nonlocal colors, num_colors
        active_colors = sorted(list(set(colors)))
        for i in range(len(active_colors)):
            for j in range(i + 1, len(active_colors)):
                c1 = active_colors[i]
                c2 = active_colors[j]
                if is_valid_merge(c1, c2, colors):
                    # Merge c2 into c1
                    for k in range(len(colors)):
                        if colors[k] == c2:
                            colors[k] = c1
                    num_colors -= 1
                    return True
        return False

    # Main search loop
    while time.time() < deadline:
        if try_merge():
            continue
        
        # If no merge possible, try random recoloring
        # Pick a random edge
        idx = random.randint(0, len(colors) - 1)
        old_color = colors[idx]
        
        # Try to change to a random other color
        # Prefer colors that already exist to avoid increasing count, 
        # or a new color if we want to explore? 
        # Actually, we want to reduce count. 
        # Changing to an existing color doesn't change count.
        # Changing to a new color increases count.
        # So we should only change to existing colors, hoping to free up old_color.
        
        active_colors = list(set(colors))
        if len(active_colors) > 1:
            new_color = random.choice([c for c in active_colors if c != old_color])
            colors[idx] = new_color
            
            # Check validity
            valid = True
            for k4 in k4s:
                k4_colors = get_k4_colors(k4, colors)
                if len(k4_colors) < 5:
                    valid = False
                    break
            
            if valid:
                # Check if old_color is still used
                if old_color not in colors:
                    num_colors -= 1
                # Keep the change
            else:
                # Revert
                colors[idx] = old_color

    return colors

# EVOLVE-BLOCK-END
