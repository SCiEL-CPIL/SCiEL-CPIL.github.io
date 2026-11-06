import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute all 4-term APs: (a, d) where a, a+d, a+2d, a+3d are all in [0, n)
    all_aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            if a + 3 * d < n:
                all_aps.append((a, d))
    
    num_aps = len(all_aps)
    
    # For each AP, store the 4 positions
    ap_positions = np.array([[a, a+d, a+2*d, a+3*d] for a, d in all_aps], dtype=np.int32)
    
    # For each position, which APs does it participate in?
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        for role in range(4):
            pos = ap_positions[idx, role]
            pos_aps[pos].append(idx)
    
    def is_valid(colors):
        """Check if the coloring is valid for all 4-term APs."""
        for idx in range(num_aps):
            vals = {colors[ap_positions[idx, 0]], colors[ap_positions[idx, 1]], 
                    colors[ap_positions[idx, 2]], colors[ap_positions[idx, 3]]}
            if len(vals) < 3:
                return False
        return True
    
    def count_colors(colors):
        return len(set(colors))
    
    def try_merge(colors, color_a, color_b):
        """Try to merge color_b into color_a. Returns new colors if valid, else None."""
        new_colors = list(colors)
        for i in range(n):
            if new_colors[i] == color_b:
                new_colors[i] = color_a
        
        # Check validity
        for idx in range(num_aps):
            vals = {new_colors[ap_positions[idx, 0]], new_colors[ap_positions[idx, 1]], 
                    new_colors[ap_positions[idx, 2]], new_colors[ap_positions[idx, 3]]}
            if len(vals) < 3:
                return None
        return new_colors
    
    def merge_pass(colors, max_iterations=50):
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
    
    def greedy_density_incremental(k_max=20, randomize=True):
        """
        Density-incremental greedy: assign positions in order, choosing color that
        minimizes the increase in critical APs (APs with <3 colors among assigned positions).
        """
        colors = [-1] * n
        # Track for each AP: how many positions assigned, and what colors are used
        ap_assigned = np.zeros(num_aps, dtype=np.int32)
        ap_colors = [set() for _ in range(num_aps)]
        
        for i in range(n):
            if time.time() > deadline:
                return None
            
            best_color = -1
            best_score = float('inf')
            
            candidate_colors = list(range(k_max))
            if randomize:
                random.shuffle(candidate_colors)
            
            for c in candidate_colors:
                # Calculate how many new critical APs this would create
                score = 0
                for ap_idx in pos_aps[i]:
                    # This AP has position i at some role
                    # Count how many positions in this AP are already assigned
                    assigned_count = ap_assigned[ap_idx]
                    if assigned_count == 0:
                        # No positions assigned yet, not critical
                        continue
                    
                    # Current colors in this AP (excluding position i)
                    current_colors = set(ap_colors[ap_idx])
                    
                    # If we assign color c to position i, new set is current_colors ∪ {c}
                    new_colors = current_colors | {c}
                    
                    # An AP is critical if it has <3 colors AND all 4 positions are assigned
                    # But we only care about APs that will become fully assigned
                    # Actually, we need to check: after assigning i, will this AP have <3 colors?
                    # The AP is at risk if the total distinct colors (including c) < 3
                    # AND all 4 positions will be assigned (which they will be eventually)
                    
                    # For now, just count if this AP would have <3 colors if all were assigned
                    # with the current assignments + this new one
                    if len(new_colors) < 3:
                        # This AP would be in danger
                        score += 1
                
                if score < best_score:
                    best_score = score
                    best_color = c
            
            if best_color == -1:
                return None
            
            colors[i] = best_color
            
            # Update AP tracking
            for ap_idx in pos_aps[i]:
                ap_assigned[ap_idx] += 1
                ap_colors[ap_idx].add(best_color)
        
        if is_valid(colors):
            return colors
        return None
    
    def greedy_with_lookahead(k_colors, lookahead=2):
        """Greedy with limited lookahead to avoid getting stuck."""
        colors = [-1] * n
        
        for i in range(n):
            if time.time() > deadline:
                return None
            
            valid_colors = []
            for c in range(k_colors):
                ok = True
                for ap_idx in pos_aps[i]:
                    vals = set()
                    for role in range(4):
                        pos = ap_positions[ap_idx, role]
                        if pos == i:
                            vals.add(c)
                        elif colors[pos] != -1:
                            vals.add(colors[pos])
                    if len(vals) < 3 and all(colors[ap_positions[ap_idx, r]] != -1 or ap_positions[ap_idx, r] == i for r in range(4)):
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
    
    best_colors = None
    best_count = n + 1
    
    # Strategy 1: Density-incremental greedy with various palettes
    for k in [6, 8, 10, 12, 16]:
        if time.time() > deadline:
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            result = greedy_density_incremental(k, randomize=True)
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
    
    # Strategy 2: Greedy with lookahead
    for k in [4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            result = greedy_with_lookahead(k, randomize=True)
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
    
    # Strategy 3: Local search / simulated annealing
    if best_colors is not None:
        current_best = list(best_colors)
        current_best_count = best_count
        
        # Try random perturbations with acceptance
        temperature = 5.0
        cooling = 0.999
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random position and try changing its color
            pos = random.randint(0, n - 1)
            old_color = current_best[pos]
            
            # Try a random new color from existing palette or a new one
            used = set(current_best)
            if random.random() < 0.7:
                new_color = random.choice(list(used))
            else:
                new_color = random.randint(0, 127)
            
            if new_color == old_color:
                continue
            
            current_best[pos] = new_color
            
            if is_valid(current_best):
                new_count = count_colors(current_best)
                delta = new_count - current_best_count
                if delta < 0:
                    current_best_count = new_count
                else:
                    # Accept with probability exp(-delta/temperature)
                    if random.random() < np.exp(-delta / temperature):
                        current_best_count = new_count
                    else:
                        current_best[pos] = old_color
            else:
                current_best[pos] = old_color
            
            temperature *= cooling
            if temperature < 0.1:
                temperature = 0.1
        
        if current_best_count < best_count:
            best_colors = current_best
            best_count = current_best_count
    
    # Final aggressive merge pass
    if best_colors is not None:
        # Multiple merge passes with random restarts on merge order
        for _ in range(10):
            if time.time() > deadline:
                break
            shuffled = list(best_colors)
            random.shuffle(shuffled)  # Shuffle doesn't change colors, but we can try different merge orders
            # Actually, let's just do merge_pass again
            final = merge_pass(best_colors)
            if count_colors(final) < best_count:
                best_colors = final
                best_count = count_colors(final)
    
    # Fallback: all distinct
    if best_colors is None:
        best_colors = list(range(n))
    
    return best_colors

