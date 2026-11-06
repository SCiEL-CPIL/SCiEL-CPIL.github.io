"""Colorings with three colors in every four-term progression."""
import random
import time

def _check_constraint(colors, a, b, c, d):
    """Check if 4-AP (a,b,c,d) uses at least 3 distinct colors."""
    ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
    return len({ca, cb, cc, cd}) >= 3

def _get_aps_involving(pos, n):
    """Get all 4-term APs (as tuples of 0-indexed positions) that involve pos."""
    aps = []
    for d in range(1, n):
        # pos as first: (pos, pos+d, pos+2d, pos+3d)
        if pos + 3*d < n:
            aps.append((pos, pos+d, pos+2*d, pos+3*d))
        # pos as second: (pos-d, pos, pos+d, pos+2d)
        if pos - d >= 0 and pos + 2*d < n:
            aps.append((pos-d, pos, pos+d, pos+2*d))
        # pos as third: (pos-2d, pos-d, pos, pos+d)
        if pos - 2*d >= 0 and pos + d < n:
            aps.append((pos-2*d, pos-d, pos, pos+d))
        # pos as fourth: (pos-3d, pos-2d, pos-d, pos)
        if pos - 3*d >= 0:
            aps.append((pos-3*d, pos-2*d, pos-d, pos))
    return aps

def _is_valid_partial(colors, n):
    """Check all complete 4-APs in the partial coloring."""
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            # All four must be colored
            if colors[a] is not None and colors[b] is not None and colors[c] is not None and colors[e] is not None:
                if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                    return False
    return True

def _try_greedy_randomized(K, n, rng, deadline):
    """Try to color with K colors using randomized greedy."""
    colors = [None] * n
    # Precompute APs for each position
    ap_cache = {}
    for pos in range(n):
        ap_cache[pos] = _get_aps_involving(pos, n)
    
    for pos in range(n):
        if time.time() > deadline:
            return None
        # Get colors that don't violate any constraint
        valid_colors = set(range(K))
        for ap in ap_cache[pos]:
            # Check if this AP is now complete
            if all(colors[p] is not None for p in ap):
                if len({colors[p] for p in ap}) < 3:
                    # This AP already violates, so no color works for pos
                    # Actually, pos is in this AP, so we need to check
                    pass
            # For APs where pos is the last uncolored position, check constraint
            uncolored = [p for p in ap if colors[p] is None]
            if len(uncolored) == 1 and uncolored[0] == pos:
                # This AP will be complete after coloring pos
                other_colors = {colors[p] for p in ap if p != pos}
                # We need len(other_colors ∪ {c}) >= 3
                # So we need: if len(other_colors) >= 3, any c works
                # if len(other_colors) == 2, c must not be in other_colors
                # if len(other_colors) <= 1, impossible (but shouldn't happen if we're careful)
                if len(other_colors) >= 3:
                    pass  # any color works
                elif len(other_colors) == 2:
                    valid_colors = valid_colors - other_colors
                else:
                    valid_colors = set()  # impossible
            elif len(uncolored) > 1:
                # AP not yet complete, no constraint yet
                pass
        
        if not valid_colors:
            return None
        
        # Pick a color, preferring ones that appear less often (to encourage diversity)
        color_counts = [0] * K
        for c in colors:
            if c is not None:
                color_counts[c] += 1
        
        # Weighted random: prefer colors with lower count
        weights = [1.0 / (1 + color_counts[c]) for c in valid_colors]
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0
        chosen = list(valid_colors)[0]
        for c, w in zip(sorted(valid_colors), weights):
            cumulative += w
            if r <= cumulative:
                chosen = c
                break
        
        colors[pos] = chosen
    
    # Final validation
    if not _is_valid_partial(colors, n):
        return None
    
    return colors

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    best_result = None
    best_K = 999
    
    # Try K from 3 upward
    for K in range(3, 15):
        if time.time() > deadline - 5:
            break
        
        # Multiple random restarts for each K
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
            
            rng = random.Random(restart * 1000 + K)
            result = _try_greedy_randomized(K, n, rng, deadline - 2)
            
            if result is not None:
                # Verify validity
                if _is_valid_partial(result, n):
                    num_colors = len(set(result))
                    if num_colors < best_K:
                        best_K = num_colors
                        best_result = result
                        # If we found 3 colors, that's optimal
                        if best_K == 3:
                            return best_result
    
    # If no result found, fall back to seed-like approach
    if best_result is None:
        colors = []
        for x in range(1, n + 1):
            for color in range(128):
                valid = True
                # Check all 4-APs ending at x
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    b = x - 2 * d
                    c = x - d
                    if a >= 1:
                        ca = colors[a - 1]
                        cb = colors[b - 1]
                        cc = colors[c - 1]
                        if len({ca, cb, cc, color}) < 3:
                            valid = False
                            break
                # Check 4-APs where x is not last
                if valid:
                    for d in range(1, (x - 1) // 2 + 1):
                        # x as second: (x-d, x, x+d, x+2d) - but x+d, x+2d may not exist yet
                        # x as third: (x-2d, x-d, x, x+d) - x+d may not exist
                        # These will be checked when later positions are colored
                        pass
                if valid:
                    colors.append(color)
                    break
    
    return best_result if best_result is not None else colors
