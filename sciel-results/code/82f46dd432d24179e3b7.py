import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 55s limit, 5s margin
    
    def check_ap_validity(colors, idx, new_color):
        """
        Returns True if assigning new_color to colors[idx] is valid 
        given colors[0..idx-1].
        """
        # Check APs where idx is the 4th term (i+3d = idx)
        max_d = idx // 3
        for d in range(1, max_d + 1):
            i = idx - 3 * d
            if i < 0:
                break
            c1 = colors[i]
            c2 = colors[i + d]
            c3 = colors[i + 2 * d]
            c4 = new_color
            if len({c1, c2, c3, c4}) < 3:
                return False
        
        # Check APs where idx is the 3rd term (i+2d = idx)
        # We need to ensure that there EXISTS a color for the 4th term (idx+d)
        # that would make the AP valid.
        max_d = idx // 2
        for d in range(1, max_d + 1):
            i = idx - 2 * d
            if i < 0:
                break
            c1 = colors[i]
            c2 = colors[i + d]
            c3 = new_color
            s3 = {c1, c2, c3}
            # If s3 has size 1, c4 must be != c1. (2 options if 3 colors)
            # If s3 has size 2, c4 must be the 3rd color. (1 option)
            # If s3 has size 3, c4 can be anything. (3 options)
            # In all cases, with 3 colors, there is at least one valid option for c4.
            # So no constraint violation here for 3 colors.
            # For >3 colors, also no violation.
            pass

        return True

    def dfs(colors, idx, num_colors, time_limit):
        if time.time() > time_limit:
            return False
        if idx == 128:
            return True
        
        # Shuffle colors to avoid bias?
        # For 3 colors, order doesn't matter much for existence.
        # For minimization, we want smallest labels.
        # But we just need ANY valid coloring with num_colors.
        # We return the first one found.
        
        for c in range(num_colors):
            if check_ap_validity(colors, idx, c):
                colors.append(c)
                if dfs(colors, idx + 1, num_colors, time_limit):
                    return True
                colors.pop()
        
        return False

    # Try 3 colors
    colors = []
    if dfs(colors, 0, 3, deadline):
        return colors

    # Try 4 colors
    colors = []
    if dfs(colors, 0, 4, deadline):
        return colors

    # Fallback: Greedy with 128 colors
    colors = []
    for idx in range(128):
        for c in range(128):
            if check_ap_validity(colors, idx, c):
                colors.append(c)
                break
        else:
            colors.append(0)
            
    return colors

# EVOLVE-BLOCK-END
