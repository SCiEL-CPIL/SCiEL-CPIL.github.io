import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute all 4-term APs: list of (a, d) where a, a+d, a+2d, a+3d are all in [0, n)
    all_aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            if a + 3 * d < n:
                all_aps.append((a, d))
    
    # For each position, which APs does it participate in?
    # pos_aps[i] = list of (ap_index, role) where role is 0,1,2,3 (position in the AP)
    pos_aps = [[] for _ in range(n)]
    for idx, (a, d) in enumerate(all_aps):
        for role in range(4):
            pos = a + role * d
            pos_aps[pos].append((idx, role))
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid(colors):
        """Check if the coloring is valid for all 4-term APs."""
        for a, d in all_aps:
            vals = {colors[a], colors[a+d], colors[a+2*d], colors[a+3*d]}
            if len(vals) < 3:
                return False
        return True
    
    def try_merge(colors, color_a, color_b):
        """Try to merge color_b into color_a. Returns new colors if valid, else None."""
        new_colors = list(colors)
        for i in range(n):
            if new_colors[i] == color_b:
                new_colors[i] = color_a
        
        # Check validity
        for a, d in all_aps:
            vals = {new_colors[a], new_colors[a+d], new_colors[a+2*d], new_colors[a+3*d]}
            if len(vals) < 3:
                return None
        return new_colors
    
    def merge_pass(colors, max_iterations=200):
        """Iteratively try to merge color pairs until no more merges possible."""
        current = list(colors)
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            used_colors = sorted(set(current))
            merged = False
            for i in range(len(used_colors)):
                for j in range(i+1, len(used_colors)):
                    if time.time() > deadline:
                        return current
                    ca, cb = used_colors[i], used_colors[j]
                    # Only merge if both colors are actually used
                    if ca not in set(current) or cb not in set(current):
                        continue
                    result = try_merge(current, ca, cb)
                    if result is not None:
                        current = result
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
        return current
    
    def greedy_construct(k_colors, randomize=True):
        """Construct a valid coloring using k colors via backtracking-lite greedy."""
        colors = [-1] * n
        
        for i in range(n):
            if time.time() > deadline:
                return None
            
            valid_colors = []
            for c in range(k_colors):
                ok = True
                # Check all 4-APs where position i is involved
                for ap_idx, role in pos_aps[i]:
                    a, d = all_aps[ap_idx]
                    # Build the set of colors in this AP
                    vals = set()
                    for r in range(4):
                        pos = a + r * d
                        if r == role:
                            vals.add(c)
                        else:
                            vals.add(colors[pos])
                    if len(vals) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            if randomize:
                colors[i] = random.choice(valid_colors)
            else:
                colors[i] = valid_colors[0]
        
        if is_valid(colors):
            return colors
        return None
    
    # Strategy 1: Start with a valid coloring using more colors, then merge
    # First, get a valid coloring with moderate number of colors
    best_colors = None
    best_count = n + 1
    
    # Try greedy with various color counts
    for k in [8, 10, 12, 16, 20]:
        if time.time() > deadline:
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            result = greedy_construct(k, randomize=True)
            if result is not None:
                cnt = count_colors(result)
                if cnt < best_count:
                    best_colors = result
                    best_count = cnt
                # Try to merge further
                merged = merge_pass(result)
                mcnt = count_colors(merged)
                if mcnt < best_count:
                    best_colors = merged
                    best_count = mcnt
    
    # Strategy 2: Try with fewer initial colors
    for k in [4, 5, 6, 7]:
        if time.time() > deadline:
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            result = greedy_construct(k, randomize=True)
            if result is not None:
                cnt = count_colors(result)
                if cnt < best_count:
                    best_colors = result
                    best_count = cnt
                merged = merge_pass(result)
                mcnt = count_colors(merged)
                if mcnt < best_count:
                    best_colors = merged
                    best_count = mcnt
    
    # Strategy 3: Random restarts with simulated annealing on color assignments
    if best_colors is None:
        # Fallback: all distinct
        best_colors = list(range(n))
    else:
        # Try to improve with random perturbations
        current_best = list(best_colors)
        current_best_count = best_count
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Pick a random position and try changing its color
            pos = random.randint(0, n - 1)
            old_color = current_best[pos]
            new_color = random.randint(0, 127)
            
            if new_color == old_color:
                continue
            
            current_best[pos] = new_color
            
            if is_valid(current_best):
                new_count = count_colors(current_best)
                if new_count < current_best_count:
                    current_best_count = new_count
                else:
                    # Revert if not improving (greedy hill climbing)
                    current_best[pos] = old_color
            else:
                current_best[pos] = old_color
        
        if current_best_count < best_count:
            best_colors = current_best
            best_count = current_best_count
    
    # Final merge pass on the best solution
    if best_colors is not None:
        final = merge_pass(best_colors)
        if count_colors(final) < best_count:
            best_colors = final
    
    return best_colors
