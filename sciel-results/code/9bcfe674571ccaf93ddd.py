import time
import numpy as np

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed a,d
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

def _progression_violated(colors, idx_set):
    # Determine if this progression is already IMPOSSIBLE to satisfy
    present = set()
    unassigned = 0
    for i in idx_set:
        c = colors[i]
        if c == -1:
            unassigned += 1
        else:
            present.add(c)
    if len(present) >= 3:
        return False  # already satisfied
    # Need to end with >=3 distinct colors
    if len(present) + unassigned < 3:
        return True  # impossible
    return False

def _check_ap_ok(ap, colors):
    """Check if AP is satisfied or still satisfiable given current colors."""
    present = set()
    unassigned = 0
    for i in ap:
        c = colors[i]
        if c == -1:
            unassigned += 1
        else:
            present.add(c)
    if len(present) >= 3:
        return True
    return len(present) + unassigned >= 3

def _greedy_with_lookahead(k_colors, deadline):
    """
    Greedy assignment with look-ahead: assign smallest valid color to each position,
    but prune choices that lead to immediate dead-ends in nearby future positions.
    """
    n = 128
    colors = [-1] * n
    
    # For each position, precompute which APs it's in
    pos_in_aps = [[] for _ in range(n)]
    for ap in _ALL_APS:
        for i in ap:
            pos_in_aps[i].append(ap)
    
    # For look-ahead: when assigning pos x, check if future positions in APs with x
    # can still be satisfied
    for x in range(n):
        if time.time() > deadline:
            return None
        
        # Get all colors that are valid for position x
        valid_colors = []
        for c in range(k_colors):
            # Tentatively assign c to x
            colors[x] = c
            
            # Check all APs containing x are still satisfiable
            ok = True
            for ap in pos_in_aps[x]:
                if not _check_ap_ok(ap, colors):
                    ok = False
                    break
            
            if not ok:
                colors[x] = -1
                continue
            
            # Look-ahead: check if any future position in APs with x becomes impossible
            # We simulate: for each future position y > x that shares an AP with x,
            # check if y can still get at least one valid color
            lookahead_ok = True
            for ap in pos_in_aps[x]:
                for y in ap:
                    if y <= x:
                        continue
                    # Check if y has at least one valid color given current assignment
                    y_has_valid = False
                    for cy in range(k_colors):
                        colors[y] = cy
                        # Check all APs containing y
                        y_ok = True
                        for ap2 in pos_in_aps[y]:
                            if not _check_ap_ok(ap2, colors):
                                y_ok = False
                                break
                        colors[y] = -1
                        if y_ok:
                            y_has_valid = True
                            break
                    if not y_has_valid:
                        lookahead_ok = False
                        break
                if not lookahead_ok:
                    break
            
            if lookahead_ok:
                valid_colors.append(c)
                # Take the first (smallest) valid color and move on
                break
            else:
                colors[x] = -1
        
        if not valid_colors:
            # Dead end - this k_colors doesn't work with greedy
            return None
        
        colors[x] = valid_colors[0]
    
    # Final verification
    for ap in _ALL_APS:
        if len({colors[i] for i in ap}) < 3:
            return None
    
    return colors

def construct():
    start = time.time()
    DEADLINE = 80.0  # safety margin before 90s limit
    
    n = 128
    
    best = None
    best_ncolors = n + 1
    
    # Try increasing number of colors with greedy + look-ahead
    for k_colors in range(3, 8):
        if time.time() > DEADLINE:
            break
        
        result = _greedy_with_lookahead(k_colors, DEADLINE)
        if result is not None:
            used = set(c for c in result if c != -1)
            ncolors = len(used)
            if ncolors < best_ncolors:
                best = result
                best_ncolors = ncolors
            break
    
    if best is not None:
        return best
    
    # Fallback: try simple patterns
    # Pattern: color by floor(log2(i+1)) mod k for various k
    for k in range(3, 10):
        if time.time() > DEADLINE:
            break
        colors = []
        for i in range(1, n+1):
            c = (i.bit_length() % k)
            colors.append(c)
        
        # Verify
        valid = True
        for ap in _ALL_APS:
            if len({colors[i] for i in ap}) < 3:
                valid = False
                break
        
        if valid:
            used = set(colors)
            if len(used) < best_ncolors:
                best = colors
                best_ncolors = len(used)
    
    if best is not None:
        return best
    
    # Ultimate fallback: unique colors (always valid)
    return list(range(n))
