import random
import time
import numpy as np

def count_violations(colors):
    n = len(colors)
    violations = 0
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            c1 = colors[i]
            c2 = colors[i + d]
            c3 = colors[i + 2 * d]
            c4 = colors[i + 3 * d]
            if len({c1, c2, c3, c4}) < 3:
                violations += 1
    return violations

def check_validity(colors):
    return count_violations(colors) == 0

def get_violation_positions(colors):
    """Return set of positions involved in violations"""
    n = len(colors)
    positions = set()
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            c1 = colors[i]
            c2 = colors[i + d]
            c3 = colors[i + 2 * d]
            c4 = colors[i + 3 * d]
            if len({c1, c2, c3, c4}) < 3:
                positions.add(i)
                positions.add(i + d)
                positions.add(i + 2 * d)
                positions.add(i + 3 * d)
    return positions

def count_violations_fast(colors_np):
    """Fast violation count using numpy broadcasting"""
    n = len(colors_np)
    violations = 0
    for d in range(1, (n - 3) // 3 + 1):
        c1 = colors_np[:n - 3*d]
        c2 = colors_np[d:n - 2*d]
        c3 = colors_np[2*d:n - d]
        c4 = colors_np[3*d:n]
        # Count distinct colors: a 4-tuple has <3 distinct if at most 2 distinct
        # This happens when: all same, or exactly 2 distinct
        # Use: number of distinct = 4 - (number of equal pairs) ... complex
        # Simpler: check if all same OR exactly 2 distinct
        all_same = (c1 == c2) & (c2 == c3) & (c3 == c4)
        # Exactly 2 distinct: at least one pair equal, and not all same
        # Actually easier: distinct < 3 means the set has size 1 or 2
        # Size 1: all equal
        # Size 2: not all equal, but at most 2 values
        # Let's just compute: for each position, check if the 4 values have <3 distinct
        # Using: sort and check
        vals = np.stack([c1, c2, c3, c4], axis=0)  # shape (4, m)
        sorted_vals = np.sort(vals, axis=0)
        # distinct < 3 means: sorted[0]==sorted[1] and sorted[2]==sorted[3] (2 distinct)
        # or all four equal (1 distinct)
        # 2 distinct: (s[0]==s[1]) & (s[2]==s[3]) & (s[1]!=s[2])
        # 1 distinct: s[0]==s[3]
        one_distinct = (sorted_vals[0] == sorted_vals[3])
        two_distinct = (sorted_vals[0] == sorted_vals[1]) & (sorted_vals[2] == sorted_vals[3]) & (sorted_vals[1] != sorted_vals[2])
        bad = one_distinct | two_distinct
        violations += np.sum(bad)
    return int(violations)

def greedy_construct():
    """Deterministic greedy: assign smallest valid color to each position"""
    colors = [-1] * 128
    for x in range(128):
        for color in range(128):
            valid = True
            for d in range(1, (127 - x) // 3 + 1):
                # Check if x is the first term: x, x+d, x+2d, x+3d
                if x + 3*d < 128:
                    c1, c2, c3, c4 = color, colors[x+d], colors[x+2*d], colors[x+3*d]
                    if len({c1, c2, c3, c4}) < 3:
                        valid = False
                        break
            if not valid:
                continue
            for d in range(1, (127 - x) // 2 + 1):
                # Check if x is second term: x-d, x, x+d, x+2d
                if x - d >= 0 and x + 2*d < 128:
                    c1, c2, c3, c4 = colors[x-d], color, colors[x+d], colors[x+2*d]
                    if len({c1, c2, c3, c4}) < 3:
                        valid = False
                        break
            if not valid:
                continue
            for d in range(1, (127 - x) // 1 + 1):
                # Check if x is third term: x-2d, x-d, x, x+d
                if x - 2*d >= 0 and x + d < 128:
                    c1, c2, c3, c4 = colors[x-2*d], colors[x-d], color, colors[x+d]
                    if len({c1, c2, c3, c4}) < 3:
                        valid = False
                        break
            if not valid:
                continue
            # x is fourth term: x-3d, x-2d, x-d, x
            for d in range(1, x // 3 + 1):
                if x - 3*d >= 0:
                    c1, c2, c3, c4 = colors[x-3*d], colors[x-2*d], colors[x-d], color
                    if len({c1, c2, c3, c4}) < 3:
                        valid = False
                        break
            if not valid:
                continue
            colors[x] = color
            break
    return colors

def try_merge_colors(colors, max_colors):
    """Try to merge two colors into one to reduce distinct count"""
    used = sorted(set(colors))
    if len(used) <= max_colors:
        return colors, len(used)
    
    best_colors = colors[:]
    best_distinct = len(used)
    
    # Try merging each pair
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            # Try merging color j into color i
            new_colors = [i if c == j else c for c in colors]
            if count_violations(new_colors) == 0:
                distinct = len(set(new_colors))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = new_colors[:]
    
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with greedy construction
    best_colors = greedy_construct()
    best_distinct = len(set(best_colors))
    
    if time.time() > deadline:
        return best_colors
    
    # Local search to improve
    colors = best_colors[:]
    
    for restart in range(1000):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize or continue from current
        if restart == 0:
            colors = best_colors[:]
        else:
            # Random perturbation
            colors = best_colors[:]
            for _ in range(random.randint(5, 20)):
                idx = random.randint(0, 127)
                colors[idx] = random.randint(0, best_distinct - 1)
        
        current_violations = count_violations(colors)
        
        for step in range(20000):
            if time.time() > deadline - 2.0:
                break
            
            if current_violations == 0:
                distinct = len(set(colors))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = colors[:]
                    # Try to merge colors further
                    merged, new_distinct = try_merge_colors(colors, best_distinct)
                    if new_distinct < best_distinct:
                        best_distinct = new_distinct
                        best_colors = merged
                break
            
            # Find positions in violations
            viol_positions = get_violation_positions(colors)
            if not viol_positions:
                break
            
            # Pick a random position from violations
            idx = random.choice(list(viol_positions))
            old_color = colors[idx]
            
            # Try changing to existing colors first
            used_colors = list(set(colors))
            random.shuffle(used_colors)
            
            improved = False
            for new_color in used_colors:
                if new_color == old_color:
                    continue
                colors[idx] = new_color
                new_violations = count_violations(colors)
                if new_violations < current_violations:
                    current_violations = new_violations
                    improved = True
                    break
                else:
                    colors[idx] = old_color
            
            if not improved:
                # Try a random new color
                new_color = random.randint(0, 127)
                if new_color != old_color:
                    colors[idx] = new_color
                    new_violations = count_violations(colors)
                    if new_violations < current_violations:
                        current_violations = new_violations
                        improved = True
                    else:
                        colors[idx] = old_color
            
            if not improved:
                # Stuck, break to restart
                break
        
        # Update best if valid
        if count_violations(colors) == 0:
            distinct = len(set(colors))
            if distinct < best_distinct:
                best_distinct = distinct
                best_colors = colors[:]
                merged, new_distinct = try_merge_colors(colors, best_distinct)
                if new_distinct < best_distinct:
                    best_distinct = new_distinct
                    best_colors = merged
    
    return best_colors
