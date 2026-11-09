import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 128
    
    # Precompute all 4-term APs: list of (a, b, c, d) 0-based indices
    # where a, b, c, d form a 4-AP: b-a = c-b = d-c = step
    # a = i, b = i+d, c = i+2d, d = i+3d, with i+3d < n
    ap_list = []
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            a = i
            b = i + d
            c = i + 2 * d
            e = i + 3 * d
            if e < n:
                ap_list.append((a, b, c, e))
    
    # For each position, which APs does it participate in?
    # For forward checking, we need to know for each position,
    # which APs it's part of, and what constraints that imposes.
    
    # pos_aps[pos] = list of (ap_index, other_positions_in_ap)
    pos_aps = defaultdict(list)
    for idx, (a, b, c, e) in enumerate(ap_list):
        for pos in (a, b, c, e):
            others = [x for x in (a, b, c, e) if x != pos]
            pos_aps[pos].append((idx, others))
    
    # Try to find coloring with min_colors colors
    def try_backtrack(min_colors):
        colors = [-1] * n
        used_colors = set()
        
        # For forward checking: for each unassigned position, track which colors are still possible
        # But this is complex. Let's do a simpler backtracking with pruning.
        
        def is_valid(pos, color):
            """Check if assigning color to pos violates any AP constraint."""
            for ap_idx, others in pos_aps[pos]:
                a, b, c, e = ap_list[ap_idx]
                # Get colors of all 4 positions
                vals = []
                for p in (a, b, c, e):
                    if p == pos:
                        vals.append(color)
                    else:
                        vals.append(colors[p])
                # All must be assigned
                if -1 in vals:
                    continue
                # Check: at least 3 distinct colors
                if len(set(vals)) < 3:
                    return False
            return True
        
        def backtrack(pos):
            if time.time() > deadline:
                return None
            if pos == n:
                return colors[:]
            
            # Try colors in order: prefer already-used colors first
            color_order = list(range(min_colors))
            # Sort: used colors first, then new colors
            color_order.sort(key=lambda c: (0 if c in used_colors else 1, c))
            
            for color in color_order:
                if not is_valid(pos, color):
                    continue
                colors[pos] = color
                new_used = set(used_colors)
                new_used.add(color)
                
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                
                colors[pos] = -1
                used_colors.discard(color)
            
            return None
        
        return backtrack(0)
    
    # Try 3 colors first
    result = try_backtrack(3)
    if result is not None:
        return result
    
    # Try 4 colors
    result = try_backtrack(4)
    if result is not None:
        return result
    
    # Try 5 colors
    result = try_backtrack(5)
    if result is not None:
        return result
    
    # Fallback: greedy with more colors
    for mc in range(6, 10):
        if time.time() > deadline:
            break
        result = try_backtrack(mc)
        if result is not None:
            return result
    
    # Emergency fallback: simple greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 3 * d < 1:
                    break
                a = x - 3 * d
                vals = [colors[a - 1], colors[a + d - 1], colors[a + 2 * d - 1], color]
                if len(set(vals)) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
        else:
            # Should not happen, but safety
            colors.append(0)
    
    return colors
