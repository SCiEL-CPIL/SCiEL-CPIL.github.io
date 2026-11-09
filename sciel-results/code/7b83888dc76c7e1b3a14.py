import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) within 1..128
    n = 128
    progressions = []
    for a in range(1, n + 1):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                progressions.append((a - 1, b - 1, c - 1, e - 1))
    
    num_progs = len(progressions)
    
    # Try to find coloring with 3 colors (0, 1, 2)
    # Use backtracking with constraint propagation
    # Key insight: for a 4-term AP to use >= 3 colors, it cannot have all 4 same,
    # and cannot have exactly 2 colors in pattern like AABA, ABAB, AABB, etc.
    # Actually it just needs >= 3 distinct values among the 4 positions.
    
    # Let's try a systematic approach: use 3 colors and search
    # with backtracking, trying to minimize backtracks
    
    best_colors = None
    best_num_colors = 4  # start with 4 colors as fallback
    
    # First, try 3 colors with backtracking
    def try_three_colors():
        nonlocal best_colors, best_num_colors
        colors = [-1] * n
        
        # Count how many positions each color appears in each progression (for pruning)
        # Use forward checking: when we assign a color to position i,
        # check all progressions containing i
        
        # For each position, store which progressions it's in and the other positions
        pos_progs = [[] for _ in range(n)]
        for pi, (a, b, c, e) in enumerate(progressions):
            for pos in (a, b, c, e):
                others = [p for p in (a, b, c, e) if p != pos]
                pos_progs[pos].append((pi, others))
        
        def is_valid_assignment(pos, color, colors):
            """Check if assigning color to pos is consistent with all progressions."""
            for (pi, others) in pos_progs[pos]:
                # Get colors of other positions (only check if all assigned)
                other_colors = []
                all_assigned = True
                for o in others:
                    if colors[o] == -1:
                        all_assigned = False
                        break
                    other_colors.append(colors[o])
                if not all_assigned:
                    continue
                # Check: {color} | other_colors must have >= 3 distinct values
                all_colors = set([color] + other_colors)
                if len(all_colors) < 3:
                    return False
            return True
        
        def backtrack(idx):
            nonlocal best_colors, best_num_colors
            if time.time() > deadline:
                return False
            if idx == n:
                # Found valid coloring with 3 colors
                best_colors = colors[:]
                best_num_colors = 3
                return True
            
            # Try colors 0, 1, 2
            for color in range(3):
                if is_valid_assignment(idx, color, colors):
                    colors[idx] = color
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
            return False
        
        # Use symmetry breaking: position 0 = 0
        colors[0] = 0
        return backtrack(1)
    
    if try_three_colors():
        return best_colors
    
    # If 3 colors fails, try 4 colors
    def try_four_colors():
        nonlocal best_colors, best_num_colors
        colors = [-1] * n
        
        pos_progs = [[] for _ in range(n)]
        for pi, (a, b, c, e) in enumerate(progressions):
            for pos in (a, b, c, e):
                others = [p for p in (a, b, c, e) if p != pos]
                pos_progs[pos].append((pi, others))
        
        def is_valid_assignment(pos, color, colors):
            for (pi, others) in pos_progs[pos]:
                other_colors = []
                all_assigned = True
                for o in others:
                    if colors[o] == -1:
                        all_assigned = False
                        break
                    other_colors.append(colors[o])
                if not all_assigned:
                    continue
                all_colors = set([color] + other_colors)
                if len(all_colors) < 3:
                    return False
            return True
        
        def backtrack(idx):
            nonlocal best_colors, best_num_colors
            if time.time() > deadline:
                return False
            if idx == n:
                best_colors = colors[:]
                best_num_colors = 4
                return True
            
            for color in range(4):
                if is_valid_assignment(idx, color, colors):
                    colors[idx] = color
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
            return False
        
        colors[0] = 0
        return backtrack(1)
    
    if try_four_colors():
        return best_colors
    
    # Fallback: use the original greedy approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    
    return colors
