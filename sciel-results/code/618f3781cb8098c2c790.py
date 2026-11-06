"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    n = 128
    
    def is_valid_color(colors, x, color):
        """Check if assigning 'color' to position x (1-indexed) is valid."""
        # Check 4-term APs ending at x: (x-3d, x-2d, x-d, x)
        max_d_4 = (x - 1) // 3
        for d in range(1, max_d_4 + 1):
            idx3 = x - 1 - 3*d
            idx2 = x - 1 - 2*d
            idx1 = x - 1 - d
            if idx3 < 0 or idx2 < 0 or idx1 < 0:
                continue
            c3 = colors[idx3]
            c2 = colors[idx2]
            c1 = colors[idx1]
            if len({c3, c2, c1, color}) < 3:
                return False
        
        # Check 3-term APs ending at x: (x-2d, x-d, x)
        max_d_3 = (x - 1) // 2
        for d in range(1, max_d_3 + 1):
            idx2 = x - 1 - 2*d
            idx1 = x - 1 - d
            if idx2 < 0 or idx1 < 0:
                continue
            c2 = colors[idx2]
            c1 = colors[idx1]
            if c2 == c1 == color:
                return False
        
        return True
    
    def greedy_coloring(prefer_existing=True):
        """Greedy coloring, optionally preferring existing colors."""
        colors = []
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            found = False
            # Try existing colors first if prefer_existing
            if prefer_existing:
                existing = list(set(colors))
                random.shuffle(existing)
                for color in existing:
                    if is_valid_color(colors, x, color):
                        colors.append(color)
                        found = True
                        break
            if not found:
                for color in range(128):
                    if is_valid_color(colors, x, color):
                        colors.append(color)
                        found = True
                        break
            if not found:
                break
        while len(colors) < n:
            colors.append(0)
        return colors[:n]
    
    # First, do a standard greedy to get a valid coloring
    best_colors = greedy_coloring(prefer_existing=False)
    best_num_colors = len(set(best_colors))
    
    # Try multiple greedy runs with preference for existing colors
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        colors = greedy_coloring(prefer_existing=True)
        num_colors = len(set(colors))
        if num_colors < best_num_colors:
            best_colors = colors
            best_num_colors = num_colors
    
    # Try local search: randomly reassign positions and see if we can reduce colors
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        colors = best_colors.copy()
        # Pick a random position to reassign
        pos = random.randint(1, n)
        old_color = colors[pos-1]
        # Try to assign a different color
        for color in range(128):
            if color == old_color:
                continue
            colors[pos-1] = color
            if is_valid_color(colors, pos, color):
                num_colors = len(set(colors))
                if num_colors < best_num_colors:
                    best_colors = colors
                    best_num_colors = num_colors
                break
        # Restore if we didn't find improvement
        if len(set(colors)) >= best_num_colors:
            colors = best_colors.copy()
    
    return best_colors
# EVOLVE-BLOCK-END
