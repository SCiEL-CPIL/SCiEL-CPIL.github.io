"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) within 0..n-1
    progressions = []
    for i in range(n):
        max_d = (n - 1 - i) // 3
        for d in range(1, max_d + 1):
            progressions.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_progs = len(progressions)
    
    # For each position, store progressions where it's the LAST term
    last_term_progs = [[] for _ in range(n)]
    for (a, b, c, e) in progressions:
        last_term_progs[e].append((a, b, c))
    
    # For each position, store progressions where it's the 3rd term (i+2d)
    third_term_progs = [[] for _ in range(n)]
    for (a, b, c, e) in progressions:
        third_term_progs[c].append((a, b, e))
    
    # For each position, store progressions where it's the 2nd term (i+d)
    second_term_progs = [[] for _ in range(n)]
    for (a, b, c, e) in progressions:
        second_term_progs[b].append((a, c, e))
    
    # For each position, store progressions where it's the 1st term (i)
    first_term_progs = [[] for _ in range(n)]
    for (a, b, c, e) in progressions:
        first_term_progs[a].append((b, c, e))
    
    def is_valid_full(colors, pos, color):
        """Check if assigning 'color' to 'pos' keeps all progressions valid where all 4 positions are assigned."""
        # Check as last term
        for (a, b, c) in last_term_progs[pos]:
            if colors[a] != -1 and colors[b] != -1 and colors[c] != -1:
                if len({colors[a], colors[b], colors[c], color}) < 3:
                    return False
        # Check as 3rd term
        for (a, b, e) in third_term_progs[pos]:
            if colors[a] != -1 and colors[b] != -1 and colors[e] != -1:
                if len({colors[a], colors[b], color, colors[e]}) < 3:
                    return False
        # Check as 2nd term
        for (a, c, e) in second_term_progs[pos]:
            if colors[a] != -1 and colors[c] != -1 and colors[e] != -1:
                if len({colors[a], color, colors[c], colors[e]}) < 3:
                    return False
        # Check as 1st term
        for (b, c, e) in first_term_progs[pos]:
            if colors[b] != -1 and colors[c] != -1 and colors[e] != -1:
                if len({color, colors[b], colors[c], colors[e]}) < 3:
                    return False
        return True
    
    def is_valid_partial(colors, pos, color):
        """
        Check partial constraints: ensure we don't create an unavoidable future violation.
        Key rule: no three consecutive terms in any AP can all have the same color,
        because then the 4th term can't make it valid.
        """
        # If pos is the 3rd term: (a, b, pos, e) - ensure a, b, pos are not all same
        for (a, b, e) in third_term_progs[pos]:
            if colors[a] == color and colors[b] == color:
                return False
        
        # If pos is the 2nd term: (a, pos, c, e) - ensure a, pos, c are not all same
        for (a, c, e) in second_term_progs[pos]:
            if colors[a] == color and colors[c] == color:
                return False
        
        # If pos is the 1st term: (pos, b, c, e) - ensure pos, b, c are not all same
        for (b, c, e) in first_term_progs[pos]:
            if colors[b] == color and colors[c] == color:
                return False
        
        return True
    
    def greedy_coloring(max_colors, seed=42):
        """Greedy coloring using colors 0..max_colors-1."""
        rng = random.Random(seed)
        colors = [-1] * n
        
        # Try multiple random orderings of colors to break symmetry
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            color_order = list(range(max_colors))
            rng.shuffle(color_order)
            
            colors = [-1] * n
            success = True
            
            for pos in range(n):
                found = False
                for color in color_order:
                    if is_valid_full(colors, pos, color) and is_valid_partial(colors, pos, color):
                        colors[pos] = color
                        found = True
                        break
                
                if not found:
                    # Try with more flexibility - relax partial constraint
                    for color in color_order:
                        if is_valid_full(colors, pos, color):
                            colors[pos] = color
                            found = True
                            break
                
                if not found:
                    success = False
                    break
            
            if success:
                # Verify full validity
                valid = True
                for (a, b, c, e) in progressions:
                    if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                        valid = False
                        break
                if valid:
                    return colors
        
        return None
    
    # Try increasing number of colors
    for k in range(3, 13):
        if time.time() > deadline - 15:
            break
        sol = greedy_coloring(k)
        if sol is not None:
            return sol
    
    # Fallback: greedy with all 128 colors, picking smallest valid
    colors = [-1] * n
    for pos in range(n):
        for color in range(128):
            if is_valid_full(colors, pos, color) and is_valid_partial(colors, pos, color):
                colors[pos] = color
                break
        if colors[pos] == -1:
            for color in range(128):
                if is_valid_full(colors, pos, color):
                    colors[pos] = color
                    break
            if colors[pos] == -1:
                colors[pos] = 0
    
    # Final verification and fix
    for (a, b, c, e) in progressions:
        if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
            # Try to fix by changing one of the positions
            for pos in (a, b, c, e):
                for new_color in range(128):
                    if new_color == colors[pos]:
                        continue
                    old = colors[pos]
                    colors[pos] = new_color
                    if is_valid_full(colors, pos, new_color):
                        # Check if this fixes the progression
                        if pos == a:
                            if len({colors[a], colors[b], colors[c], colors[e]}) >= 3:
                                break
                        elif pos == b:
                            if len({colors[a], colors[b], colors[c], colors[e]}) >= 3:
                                break
                        elif pos == c:
                            if len({colors[a], colors[b], colors[c], colors[e]}) >= 3:
                                break
                        elif pos == e:
                            if len({colors[a], colors[b], colors[c], colors[e]}) >= 3:
                                break
                else:
                    colors[pos] = old
                    continue
                break
    
    return colors

# EVOLVE-BLOCK-END
