import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    # Precompute all 4-term APs: list of (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # For each position, store indices of APs it participates in
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def count_violations_fast(colors):
        """Count violations using precomputed APs."""
        violations = 0
        for ap in aps:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c1, c2, c3, c4}) < 3:
                violations += 1
        return violations
    
    def check_validity_fast(colors):
        for ap in aps:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c1, c2, c3, c4}) < 3:
                return False
        return True
    
    def try_merge_colors(colors, max_colors):
        """Try to merge two colors to reduce distinct count while maintaining validity."""
        distinct_colors = list(set(colors))
        if len(distinct_colors) <= 2:
            return colors
        
        best_merged = None
        best_violations = count_violations_fast(colors)
        
        # Try merging each pair of colors
        for i in range(len(distinct_colors)):
            for j in range(i + 1, len(distinct_colors)):
                c1, c2 = distinct_colors[i], distinct_colors[j]
                
                # Create merged coloring
                merged = [c1 if c == c2 else c for c in colors]
                v = count_violations_fast(merged)
                
                if v == 0:
                    return merged
                elif best_merged is None or v < best_violations:
                    best_violations = v
                    best_merged = merged
        
        return best_merged if best_merged is not None else colors
    
    best_colors = None
    best_distinct = 129
    
    # Phase 1: Find valid colorings with simulated annealing
    for max_colors in range(3, 8):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts with different strategies
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random colors
            colors = [random.randint(0, max_colors - 1) for _ in range(n)]
            
            # Simulated annealing
            T = 5.0
            cooling = 0.995
            min_T = 0.01
            
            for step in range(3000):
                if time.time() > deadline - 10:
                    break
                
                if T < min_T:
                    break
                
                # Pick a random position
                idx = random.randint(0, n - 1)
                old_color = colors[idx]
                new_color = random.randint(0, max_colors - 1)
                
                if new_color == old_color:
                    continue
                
                # Compute violation delta for this position change
                old_violations = 0
                new_violations = 0
                
                for ap_idx in pos_aps[idx]:
                    ap = aps[ap_idx]
                    # Check old coloring
                    old_c = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                    old_c[ap.index(idx)] = old_color  # Ensure old color at idx
                    if len(set(old_c)) < 3:
                        old_violations += 1
                    
                    # Check new coloring
                    new_c = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                    new_c[ap.index(idx)] = new_color  # Set new color at idx
                    if len(set(new_c)) < 3:
                        new_violations += 1
                
                delta = new_violations - old_violations
                
                # Accept if improvement or with probability exp(-delta/T)
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    colors[idx] = new_color
                    T *= cooling
                else:
                    T *= cooling
            
            # Check if valid
            if check_validity_fast(colors):
                distinct = len(set(colors))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = colors.copy()
                
                # Try to merge colors to reduce distinct count
                merged = try_merge_colors(colors, max_colors)
                if check_validity_fast(merged):
                    merged_distinct = len(set(merged))
                    if merged_distinct < best_distinct:
                        best_distinct = merged_distinct
                        best_colors = merged.copy()
    
    # Phase 2: Try to further reduce colors from best found
    if best_colors is not None and time.time() < deadline - 5:
        current = best_colors.copy()
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            current = try_merge_colors(current, 128)
            if check_validity_fast(current):
                if len(set(current)) < best_distinct:
                    best_distinct = len(set(current))
                    best_colors = current.copy()
    
    if best_colors is not None:
        return best_colors
    else:
        # Fallback: Greedy construction
        colors = []
        for x in range(1, n + 1):
            placed = False
            for color in range(128):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    if x - 1 - 3 * d >= 0:
                        c1 = colors[x - 1 - 3 * d]
                        c2 = colors[x - 1 - 2 * d]
                        c3 = colors[x - 1 - d]
                        c4 = color
                        if len({c1, c2, c3, c4}) < 3:
                            valid = False
                            break
                if valid:
                    colors.append(color)
                    placed = True
                    break
            if not placed:
                colors.append(127)
        return colors
