import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 128
    
    # Precompute all 4-term APs and 3-term APs for fast checking
    # 4-term AP: (i, i+d, i+2d, i+3d)
    aps4 = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            aps4.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # 3-term AP: (i, i+d, i+2d)
    aps3 = []
    for d in range(1, (N - 2) // 2 + 1):
        for i in range(0, N - 2 * d):
            aps3.append((i, i + d, i + 2 * d))
    
    # For each position, precompute which 4-term APs it participates in as the last element
    # and which 3-term APs it participates in as the last element
    pos_aps4_last = [[] for _ in range(N)]  # pos_aps4_last[pos] = list of (d, pos-d, pos-2d, pos-3d)
    pos_aps3_last = [[] for _ in range(N)]  # pos_aps3_last[pos] = list of (d, pos-d, pos-2d)
    
    for (a, b, c, d_pos) in aps4:
        pos_aps4_last[d_pos].append((a, b, c))
    
    for (a, b, c) in aps3:
        pos_aps3_last[c].append((a, b))
    
    def is_valid_assignment(colors, pos, color):
        """Check if assigning 'color' to position 'pos' is valid given current colors."""
        # Check 4-term APs where pos is the last element
        for (a, b, c) in pos_aps4_last[pos]:
            if colors[a] == -1 or colors[b] == -1 or colors[c] == -1:
                continue  # not yet assigned, skip
            distinct = len({color, colors[a], colors[b], colors[c]})
            if distinct < 3:
                return False
        # Check 3-term APs where pos is the last element
        for (a, b) in pos_aps3_last[pos]:
            if colors[a] == -1 or colors[b] == -1:
                continue
            if colors[a] == color and colors[b] == color:
                return False
        return True
    
    def try_assign(num_colors, max_restarts=100):
        """Try to assign using exactly num_colors colors (0 to num_colors-1)."""
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                return None
            
            colors = [-1] * N
            colors[0] = random.randint(0, num_colors - 1)
            colors[1] = random.randint(0, num_colors - 1)
            colors[2] = random.randint(0, num_colors - 1)
            
            success = True
            for pos in range(3, N):
                if time.time() - start_time > DEADLINE:
                    return None
                
                # Try colors in random order
                color_order = list(range(num_colors))
                random.shuffle(color_order)
                
                assigned = False
                for color in color_order:
                    if is_valid_assignment(colors, pos, color):
                        colors[pos] = color
                        assigned = True
                        break
                
                if not assigned:
                    # Backtrack: try a different assignment for a previous position
                    # Simple backtracking: find the most recent position we can change
                    found_backtrack = False
                    for back_pos in range(pos - 1, 2, -1):
                        if time.time() - start_time > DEADLINE:
                            return None
                        
                        # Try different colors for back_pos
                        old_color = colors[back_pos]
                        color_order = list(range(num_colors))
                        random.shuffle(color_order)
                        # Remove old color from consideration first
                        color_order.remove(old_color)
                        
                        for new_color in color_order:
                            if is_valid_assignment(colors, back_pos, new_color):
                                colors[back_pos] = new_color
                                # Re-assign positions after back_pos
                                for reset_pos in range(back_pos + 1, pos):
                                    colors[reset_pos] = -1
                                
                                # Re-assign from back_pos+1 to pos-1
                                reassign_success = True
                                for rp in range(back_pos + 1, pos):
                                    if time.time() - start_time > DEADLINE:
                                        return None
                                    color_order2 = list(range(num_colors))
                                    random.shuffle(color_order2)
                                    ra = False
                                    for rc in color_order2:
                                        if is_valid_assignment(colors, rp, rc):
                                            colors[rp] = rc
                                            ra = True
                                            break
                                    if not ra:
                                        reassign_success = False
                                        break
                                
                                if reassign_success:
                                    # Now try to assign pos
                                    for color in color_order:
                                        if is_valid_assignment(colors, pos, color):
                                            colors[pos] = color
                                            found_backtrack = True
                                            break
                                    if found_backtrack:
                                        break
                        
                        if found_backtrack:
                            break
                        colors[back_pos] = old_color  # restore
                    
                    if not found_backtrack:
                        success = False
                        break
            
            if success:
                # Verify the solution
                for (a, b, c, d_pos) in aps4:
                    distinct = len({colors[a], colors[b], colors[c], colors[d_pos]})
                    if distinct < 3:
                        success = False
                        break
                if success:
                    return colors
        
        return None
    
    # Try with 3 colors first, then 4, etc.
    for num_colors in [3, 4, 5, 6, 7, 8]:
        if time.time() - start_time > DEADLINE:
            break
        
        result = try_assign(num_colors)
        if result is not None:
            # Map to 0..127 range (already in 0..num_colors-1 which is within 0..127)
            return result
    
    # Fallback: use the original greedy approach with more colors
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                   for d in range(1, (x-1)//3 + 1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                       for d in range(1, (x-1)//2 + 1))):
                colors.append(color)
                break
    return colors
# EVOLVE-BLOCK-END
