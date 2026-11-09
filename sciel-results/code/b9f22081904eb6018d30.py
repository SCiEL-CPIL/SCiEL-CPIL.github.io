"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    n = 128
    
    # Precompute all 4-term APs: list of (i, i+d, i+2d, i+3d)
    aps = []
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            a = i
            b = i + d
            c = i + 2 * d
            e = i + 3 * d
            if e < n:
                aps.append((a, b, c, e))
    
    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[e].append(idx)
    
    # Try with 3 colors first, then 4
    for num_colors in [3, 4]:
        colors = [-1] * n
        colors[0] = 0
        
        def is_valid(pos, color):
            """Check if assigning 'color' to position 'pos' violates any constraint."""
            for ap_idx in pos_aps[pos]:
                a, b, c, e = aps[ap_idx]
                # Get current colors for the 4 positions
                col_a = colors[a]
                col_b = colors[b]
                col_c = colors[c]
                col_e = colors[e]
                
                # If any is unassigned, skip (will be checked later)
                if -1 in (col_a, col_b, col_c, col_e):
                    continue
                
                # Check: at least 3 distinct colors
                distinct = len({col_a, col_b, col_c, col_e})
                if distinct < 3:
                    return False
            return True
        
        # Greedy with backtracking
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            # Try colors in order
            for color in range(num_colors):
                if is_valid(pos, color):
                    colors[pos] = color
                    if backtrack(pos + 1):
                        return True
                    colors[pos] = -1
            return False
        
        # Use a smarter greedy first
        def greedy():
            if time.time() > deadline:
                return False
            for pos in range(1, n):
                found = False
                for color in range(num_colors):
                    if is_valid(pos, color):
                        colors[pos] = color
                        found = True
                        break
                if not found:
                    return False
            return True
        
        if greedy():
            # Verify
            valid = True
            for a, b, c, e in aps:
                if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                    valid = False
                    break
            if valid:
                return colors
        
        # If greedy failed, try backtracking (limited)
        colors = [-1] * n
        colors[0] = 0
        if backtrack(1):
            # Verify
            valid = True
            for a, b, c, e in aps:
                if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                    valid = False
                    break
            if valid:
                return colors
    
    # Fallback: use the original greedy with many colors
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
